"""
pose_detector.py

Enhanced version: detects walking behavior more accurately
by combining knee angle motion + body displacement over time.
"""
import cv2
import mediapipe as mp
import numpy as np
import math


class WalkingPoseDetector:
    """
    Detects walking by analyzing both:
      1. Temporal variation of knee angles (rhythmic bending)
      2. Horizontal body movement across frames
    """

    def __init__(
        self,
        knee_angle_thresh=160.0,
        history_length=5,
        angle_variation_thresh=10.0,
        movement_thresh=15,
    ):
        self.knee_angle_thresh = float(knee_angle_thresh)
        self.history_length = int(history_length)
        self.angle_variation_thresh = float(angle_variation_thresh)
        self.movement_thresh = int(movement_thresh)

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # higher accuracy
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        # Internal state
        self.walking_alerts = set()
        self.motion_history = {}  # track_id -> list of (left_angle, right_angle)
        self.track_positions = {}  # track_id -> list of centroid x positions

    def _calculate_angle(self, a, b, c):
        """Calculates the angle at point b (in degrees)."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def update(self, frame, tracks):
        """
        Processes the frame and tracks to detect walking.

        Args:
            frame: The current video frame (BGR).
            tracks: A list of active DeepSORT track objects.

        Returns:
            A set of track IDs for people who are detected as walking.
        """
        self.walking_alerts.clear()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            # Crop validation
            if l < 0 or t < 0 or r > frame.shape[1] or b > frame.shape[0]:
                continue
            person_crop = frame[t:b, l:r]
            if person_crop.size == 0:
                continue

            image_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if not results.pose_landmarks:
                continue

            landmarks = results.pose_landmarks.landmark
            mp_pose = mp.solutions.pose.PoseLandmark

            # Extract keypoints
            left_hip = [landmarks[mp_pose.LEFT_HIP.value].x, landmarks[mp_pose.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.LEFT_KNEE.value].x, landmarks[mp_pose.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.LEFT_ANKLE.value].x, landmarks[mp_pose.LEFT_ANKLE.value].y]

            right_hip = [landmarks[mp_pose.RIGHT_HIP.value].x, landmarks[mp_pose.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.RIGHT_KNEE.value].x, landmarks[mp_pose.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.RIGHT_ANKLE.value].x, landmarks[mp_pose.RIGHT_ANKLE.value].y]

            # Compute angles
            left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)

            # Store history
            angles = self.motion_history.setdefault(track_id, [])
            angles.append((left_knee_angle, right_knee_angle))
            if len(angles) > self.history_length:
                angles.pop(0)

            # Calculate variation
            left_var = np.std([a[0] for a in angles])
            right_var = np.std([a[1] for a in angles])

            # Track horizontal motion
            cx = (l + r) // 2
            pos_list = self.track_positions.setdefault(track_id, [])
            pos_list.append(cx)
            if len(pos_list) > self.history_length:
                pos_list.pop(0)

            moving_horizontally = (
                len(pos_list) >= self.history_length
                and abs(pos_list[-1] - pos_list[0]) > self.movement_thresh
            )

            # Final walking logic: must be moving + knees bending rhythmically
            if moving_horizontally and (
                left_var > self.angle_variation_thresh or right_var > self.angle_variation_thresh
            ):
                self.walking_alerts.add(track_id)

        return self.walking_alerts

    def close(self):
        self.pose.close()
