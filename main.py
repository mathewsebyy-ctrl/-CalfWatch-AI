from calf_detection import detect_calf
from behaviour_detection import detect_behavior
from risk_engine import calculate_risk


def main():

    print("================================")
    print("       CALFWATCH AI")
    print("================================")

    # Temporary frame
    frame = None

    # Step 1: Calf detection
    calf_detected = detect_calf(frame)

    print("\nCalf detected:", calf_detected)

    if not calf_detected:
        print("No calf detected.")
        return

    # Step 2: Behaviour detection
    behavior = detect_behavior(frame, calf_detected)

    print("Behaviour detected:", behavior)

    # Temporary timing values
    time_to_stand = 40
    time_to_suckle = 90

    print("Time to stand:", time_to_stand, "seconds")
    print("Time to first suckle:", time_to_suckle, "seconds")

    # Step 3: Risk calculation
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