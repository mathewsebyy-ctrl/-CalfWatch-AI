time_to_stand = None
time_to_suckle = None
current_behavior = "Unknown"
previous_behavior = None

# Track calf movement for basic suckling estimation
previous_bbox = None
movement_count = 0

def detect_behavior(frame, calf_result, elapsed_time):
    global time_to_stand
    global time_to_suckle
    global current_behavior
    global previous_behavior
    global previous_bbox
    global movement_count

    # No calf detected
    if not calf_result["detected"]:
        current_behavior = "No Calf"
        return current_behavior

    # Get bounding box
    x1, y1, x2, y2 = calf_result["bbox"]

    width = x2 - x1
    height = y2 - y1

    # Basic posture estimation
    if width > height:
        current_behavior = "Lying"
    else:
        current_behavior = "Standing"

    # Detect first Lying -> Standing transition
    if (
        current_behavior == "Standing"
        and previous_behavior == "Lying"
        and time_to_stand is None
    ):
        time_to_stand = round(elapsed_time, 2)

    # ------------------------------------------------
    # Basic movement-based suckling estimation
    # ------------------------------------------------

    if previous_bbox is not None:

        old_x1, old_y1, old_x2, old_y2 = previous_bbox

        old_center_x = (old_x1 + old_x2) / 2
        old_center_y = (old_y1 + old_y2) / 2

        new_center_x = (x1 + x2) / 2
        new_center_y = (y1 + y2) / 2

        movement = (
            abs(new_center_x - old_center_x)
            + abs(new_center_y - old_center_y)
        )

        # Small repeated movement can indicate feeding activity
        if movement > 2:
            movement_count += 1

        # Prototype threshold
        if (
            movement_count >= 10
            and time_to_suckle is None
            and time_to_stand is not None
            and elapsed_time > time_to_stand
        ):
            time_to_suckle = round(elapsed_time, 2)

    previous_bbox = (x1, y1, x2, y2)

    # Remember previous behaviour
    if current_behavior != "No Calf":
        previous_behavior = current_behavior

    return current_behavior