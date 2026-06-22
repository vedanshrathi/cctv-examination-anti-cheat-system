# Suspicious Behavior Detection in Exam Halls

## Overview

This project is an AI-powered exam hall monitoring system that detects suspicious student behavior using computer vision and deep learning techniques. The system combines YOLOv8 object detection, DeepSORT tracking, MediaPipe pose estimation, and custom behavior analysis algorithms to identify activities that may indicate cheating or misconduct during examinations.

## Features

### Person Detection and Tracking

* Detects students in video streams using YOLOv8.
* Tracks individuals across frames using DeepSORT.
* Assigns unique IDs to each detected student.

### Suspicious Behavior Detection

* **Looking Away Detection**

  * Identifies students who repeatedly look away from their workspace.
  * Uses bounding-box aspect ratio analysis over time.

* **Walking Detection**

  * Detects students moving around the exam hall.
  * Combines pose estimation and body displacement analysis.

* **Interaction Detection**

  * Identifies prolonged close proximity between students.
  * Flags potential communication or collaboration.

### Automatic Zone Detection

* Automatically generates monitoring zones around student-bench pairs.
* Highlights zones where suspicious activities occur.

### Alert Logging

* Logs all detected suspicious events into a CSV file.
* Records:

  * Timestamp
  * Student ID
  * Alert Type
  * Additional Information

## Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* DeepSORT Realtime
* MediaPipe Pose
* NumPy

## Project Structure

```text
.
├── main.py
├── behaviour_detector.py
├── pose_detector.py
├── zone_autodetector.py
├── utils.py
├── requirements.txt
├── alerts.csv
└── README.md
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/exam-hall-behaviour-detection.git
cd exam-hall-behaviour-detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download YOLOv8 Model

The project uses YOLOv8. The model will be downloaded automatically by Ultralytics if not already present.

## Usage

Basic execution:

```bash
python main.py --video exam_video.mp4 --display --auto-zone
```

### Example

```bash
python main.py \
    --video exam_hall.mp4 \
    --display \
    --auto-zone \
    --conf 0.5
```

## Command Line Arguments

| Argument           | Description                                  |
| ------------------ | -------------------------------------------- |
| --video            | Input video path                             |
| --display          | Show live detection window                   |
| --auto-zone        | Automatically generate monitoring zones      |
| --yolo             | Path to YOLO model                           |
| --conf             | Detection confidence threshold               |
| --knee-angle       | Walking detection knee-angle threshold       |
| --walking-history  | Number of frames for walking analysis        |
| --angle-var        | Knee angle variation threshold               |
| --movement-thresh  | Horizontal movement threshold                |
| --interaction-dist | Distance threshold for interaction detection |
| --interaction-time | Duration required for interaction alert      |
| --away-time        | Looking-away detection duration              |
| --away-ratio       | Looking-away aspect-ratio threshold          |

## Output

### Visual Output

* Bounding boxes around students.
* Unique track IDs.
* Alert labels:

  * Looking Away
  * Walking
  * Interaction

### CSV Log Output

Example:

```csv
Timestamp_sec,Track_ID,Alert_Type,Info
12.50,3,Walking,
18.23,5,Interaction,with 7
24.67,2,Looking Away,
```

## Future Improvements

* Head pose estimation for more accurate gaze tracking.
* Cheating object detection (phones, notes, books).
* Real-time CCTV integration.
* Multi-camera support.
* Dashboard for invigilators.
* Email/SMS alert notifications.

## Applications

* Examination Monitoring
* Academic Integrity Systems
* Classroom Surveillance
* Automated Proctoring Solutions

## Author

Rishabh Chaudhari & Vedansh Rathi

## License

This project is intended for educational and research purposes.
