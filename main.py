import cv2
import behaviour_detection
from calf_detection import detect_calf
from risk_engine import calculate_risk

VIDEO_PATH = "data/test_videos/calf_test.mp4.mp4"

def main():
    print("================================")
    print("       CALFWATCH AI")
    print("================================")

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Could not open video at: {VIDEO_PATH}")
        return
    
    print("Video loaded successfully.\n")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30.0

    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1
        elapsed_time = frame_number / fps

        # 1. Detect Calf
        calf_result = detect_calf(frame)

        # 2. Detect Behaviour using Albin's original function
        behavior = behaviour_detection.detect_behavior(
            frame,
            calf_result,
            elapsed_time
        )

        # Print update every 30 frames (approx 1 second)
        if frame_number % 30 == 0:
            print(f"Time: {elapsed_time:.1f}s | Behaviour: {behavior}")

    cap.release()

    print("\n================================")
    print("       FINAL RESULTS")
    print("================================")

    # 3. Retrieve global variables updated by Albin's module
    time_to_stand = behaviour_detection.time_to_stand
    time_to_suckle = behaviour_detection.time_to_suckle

    print("Time to stand:", time_to_stand, "seconds")
    print("Time to first suckle:", time_to_suckle, "seconds")

    # 4. Calculate Final Risk Score
    risk, score = calculate_risk(
        time_to_stand,
        time_to_suckle
    )

    print("\nRisk score:", score)
    print("Risk level:", risk)

    print("\n================================")
    print("       PIPELINE COMPLETE")
    print("================================")


if __name__ == "__main__":
    main()