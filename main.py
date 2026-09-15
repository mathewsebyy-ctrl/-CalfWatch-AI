import cv2
import json
import behaviour_detection
from calf_detection import detect_calf
from risk_engine import calculate_risk

# Note: You had ".mp4.mp4" in your original path. I've left it as-is, 
# but you may want to check if it should just be ".mp4"
VIDEO_PATH = "data/test_videos/calf_test.mp4.mp4" 
STATUS_PATH = "data/status.json"

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
    latest_behavior = "Unknown"
    calf_detected = False

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1
        elapsed_time = frame_number / fps

        # 1. Detect calf
        calf_result = detect_calf(frame)
        calf_detected = calf_result["detected"]

        # 2. Detect behaviour
        behavior = behaviour_detection.detect_behavior(
            frame,
            calf_result,
            elapsed_time
        )

        latest_behavior = behavior

        # Print update every 30 frames
        if frame_number % 30 == 0:
            print(
                f"Time: {elapsed_time:.1f}s | "
                f"Behaviour: {behavior}"
            )

    cap.release()

    # 3. Retrieve final timing results
    time_to_stand = behaviour_detection.time_to_stand
    time_to_suckle = behaviour_detection.time_to_suckle

    # 4. Calculate risk
    risk, score = calculate_risk(
        time_to_stand,
        time_to_suckle
    )

    # 5. Create dashboard status
    status = {
        "calf_detected": calf_detected,
        "current_behavior": latest_behavior,
        "monitoring_status": "Completed",
        "time_to_stand": time_to_stand,
        "time_to_suckle": time_to_suckle,
        "risk_score": score,
        "risk_level": risk
    }

    # 6. Save results for Streamlit
    with open(STATUS_PATH, "w") as file:
        json.dump(status, file, indent=4)

    print("\n================================")
    print("       FINAL RESULTS")
    print("================================")

    print("Time to stand:", time_to_stand, "seconds")
    print("Time to first suckle:", time_to_suckle, "seconds")
    print("Risk score:", score)
    print("Risk level:", risk)

    print("\nDashboard status saved to:")
    print(STATUS_PATH)

    print("\n================================")
    print("       PIPELINE COMPLETE")
    print("================================")


if __name__ == "__main__":
    main()