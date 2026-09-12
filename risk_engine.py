def calculate_risk(time_to_stand, time_to_suckle):
    """
    Prototype risk calculation.

    NOTE:
    These thresholds are only for the SIH prototype.
    They are NOT clinical veterinary thresholds.
    """

    risk_score = 0

    # Prototype threshold for standing
    if time_to_stand is None:
        risk_score += 2
    elif time_to_stand > 60:
        risk_score += 2
    elif time_to_stand > 30:
        risk_score += 1

    # Prototype threshold for suckling
    if time_to_suckle is None:
        risk_score += 2
    elif time_to_suckle > 120:
        risk_score += 2
    elif time_to_suckle > 60:
        risk_score += 1

    # Convert score into risk category
    if risk_score >= 3:
        risk = "HIGH"
    elif risk_score >= 1:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return risk, risk_score