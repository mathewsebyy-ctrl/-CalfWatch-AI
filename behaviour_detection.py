def detect_behavior(frame, calf_detected):
    """
    Temporary behaviour detection function.

    This will later be replaced with Albin's
    actual behaviour detection module.
    """

    if not calf_detected:
        return "NOT_DETECTED"

    # Temporary behaviour for integration testing
    return "STANDING"