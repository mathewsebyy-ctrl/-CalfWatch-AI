import cv2
import behaviour_detection
from calf_detection import detect_calf
from risk_engine import calculate_risk

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    # Reset previous analysis results
    behaviour_detection.time_to_stand = None
    behaviour_detection.time_to_suckle = None
    behaviour_detection.current_behavior = "Unknown"
    behaviour_detection.previous_behavior = None
    behaviour_detection.previous_bbox = None
    behaviour_detection.movement_count = 0

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30.0

    frame_number = 0
    latest_behavior = "Unknown"
    calf_detected = False

    while True:
        # These lines must be indented inside the while loop
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Analyze every 3rd frame to improve speed
        if frame_number % 2 != 0:
            continue

        # Calculate elapsed time ONLY for the frames being analyzed
        elapsed_time = frame_number / fps

        calf_result = detect_calf(frame)
        calf_detected = calf_result["detected"]

        # Behaviour detection
        latest_behavior = behaviour_detection.detect_behavior(
            frame,
            calf_result,
            elapsed_time
        )

    cap.release()

    time_to_stand = behaviour_detection.time_to_stand
    time_to_suckle = behaviour_detection.time_to_suckle

    risk, score = calculate_risk(
        time_to_stand,
        time_to_suckle
    )

    return {
        "calf_detected": calf_detected,
        "current_behavior": latest_behavior,
        "monitoring_status": "Completed",
        "time_to_stand": time_to_stand,
        "time_to_suckle": time_to_suckle,
        "risk_score": score,
        "risk_level": risk
    }