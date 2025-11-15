#!/usr/bin/env python3
"""
main.py
Suspicious Behavior Detection for Exam Halls using YOLOv8 + DeepSORT
"""

import argparse
import cv2
import numpy as np
import csv
from pathlib import Path

from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from utils import draw_track_box
from behaviour_detector import LookAwayDetector, InteractionDetector
from zone_autodetector import create_zones_from_frame
from pose_detector import WalkingPoseDetector  # Updated detector


class AlertLogger:
    """Logs alerts to a CSV file."""
    def __init__(self, filepath="alerts.csv"):
        self.filepath = Path(filepath)
        self.file = self.filepath.open(mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Timestamp_sec", "Track_ID", "Alert_Type", "Info"])

    def log(self, timestamp, track_id, alert_type, info=""):
        self.writer.writerow([f"{timestamp:.2f}", track_id, alert_type, info])
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


def draw_bench_zones(frame, zones, alerting_zone_indices):
    """Draws static bench zones, coloring them based on alerts."""
    for i, zone in enumerate(zones):
        color = (0, 0, 255) if i in alerting_zone_indices else (0, 255, 0)
        cv2.polylines(frame, [zone], True, color, 2)


def parse_args():
    p = argparse.ArgumentParser(description="Suspicious Behavior Detection")
    p.add_argument("--video", type=str, required=True, help="Path to input video")
    p.add_argument("--display", action="store_true", help="Show live window")
    p.add_argument("--auto-zone", action="store_true", help="Automatically detect bench zones on the first frame")
    p.add_argument("--yolo", type=str, default="yolov8n.pt", help="Path to YOLOv8 model (.pt)")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for YOLO object detection")

    # --- Arguments for tuning detectors ---
    p.add_argument("--knee-angle", type=float, default=160.0,
                   help="Knee angle threshold to trigger 'Walking' alert. Lower is stricter.")
    p.add_argument("--walking-history", type=int, default=5,
                   help="Number of frames to analyze for consistent walking motion.")
    p.add_argument("--angle-var", type=float, default=10.0,
                   help="Minimum knee angle variation across frames to classify walking.")
    p.add_argument("--movement-thresh", type=int, default=15,
                   help="Horizontal pixel movement threshold for walking detection.")

    p.add_argument("--interaction-dist", type=float, default=100.0, help="Pixel gap to trigger 'Interaction' alert")
    p.add_argument("--interaction-time", type=float, default=2.0, help="Seconds of proximity for 'Interaction' alert")
    p.add_argument("--away-time", type=float, default=3.0, help="Seconds to trigger 'Looking Away' alert")
    p.add_argument("--away-ratio", type=float, default=0.6, help="Aspect ratio threshold for looking away")

    return p.parse_args()


def main():
    args = parse_args()
    yolo_net = YOLO(args.yolo)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {args.video}")

    zones = []
    if args.auto_zone:
        ret, first_frame = cap.read()
        if not ret:
            raise IOError("Cannot read the first frame of the video for auto-zoning.")
        zones = create_zones_from_frame(first_frame, yolo_net)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        print("[WARN] --auto-zone not specified. Bench zone-related alerts will be disabled.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = int(1000 / fps) if args.display else 1
    frame_idx = 0

    tracker = DeepSort(max_age=20, n_init=5)

    # Initialize detectors
    look_away_detector = LookAwayDetector(
        away_threshold=args.away_ratio,
        time_window_sec=args.away_time,
        fps=fps,
    )
    interaction_detector = InteractionDetector(
        proximity_threshold=args.interaction_dist,
        time_threshold_sec=args.interaction_time,
    )
    walking_detector = WalkingPoseDetector(
        knee_angle_thresh=args.knee_angle,
        history_length=args.walking_history,
        angle_variation_thresh=args.angle_var,
        movement_thresh=args.movement_thresh,
    )

    alert_logger = AlertLogger()
    student_states = {}
    track_zone_mapping = {}

    print("[INFO] Processing video... Press 'q' in the display window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            current_time = frame_idx / fps

            results = yolo_net(frame, verbose=False, classes=[0])[0]
            detections = []
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > args.conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, "person"))

            tracks = tracker.update_tracks(detections, frame=frame)

            walking_alerts = walking_detector.update(frame, tracks)
            interaction_alerts = interaction_detector.update(tracks, current_time)

            alerting_zone_indices = set()

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()

                if track_id not in student_states:
                    student_states[track_id] = {'alerts': set(), 'last_logged': {}}

                current_alerts = set()

                # Zone tracking
                point = (int((ltrb[0] + ltrb[2]) / 2), int(ltrb[3]))
                for i, zone in enumerate(zones):
                    if cv2.pointPolygonTest(zone, point, False) >= 0:
                        track_zone_mapping[track_id] = i
                        break

                if look_away_detector.update(track_id, ltrb, current_time):
                    current_alerts.add("Looking Away")

                if track_id in walking_alerts:
                    current_alerts.add("Walking")
                    if not student_states[track_id]['last_logged'].get("Walking"):
                        alert_logger.log(current_time, track_id, "Walking")
                        student_states[track_id]['last_logged']["Walking"] = True
                else:
                    student_states[track_id]['last_logged']["Walking"] = False

                for id1, id2 in interaction_alerts:
                    if track_id in [id1, id2]:
                        current_alerts.add("Interaction")
                        if track_id == id1:
                            alert_logger.log(current_time, track_id, "Interaction", f"with {id2}")

                student_states[track_id]['alerts'] = current_alerts

                if current_alerts and track_id in track_zone_mapping:
                    alerting_zone_indices.add(track_zone_mapping[track_id])

                if args.display:
                    draw_track_box(frame, ltrb, track_id, alerts=current_alerts)

            # Highlight zones with walking or interaction
            for track_id in walking_alerts:
                if track_id in track_zone_mapping:
                    alerting_zone_indices.add(track_zone_mapping[track_id])

            for id1, id2 in interaction_alerts:
                if id1 in track_zone_mapping:
                    alerting_zone_indices.add(track_zone_mapping[id1])
                if id2 in track_zone_mapping:
                    alerting_zone_indices.add(track_zone_mapping[id2])

            if args.display:
                draw_bench_zones(frame, zones, alerting_zone_indices)
                cv2.imshow("Suspicious Behavior Detection", frame)
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        alert_logger.close()
        walking_detector.close()
        if args.display:
            cv2.destroyAllWindows()
        print("[INFO] Finished processing. Alerts have been logged to alerts.csv")


if __name__ == "__main__":
    main()
