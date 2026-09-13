"""
detection/calf_detection.py
=============================
CALF DETECTION MODULE (Jiwin's original responsibility)

Detects calves in a video frame using a PRETRAINED Ultralytics YOLOv8
model (trained on COCO). No training required.

WHY "COW" CLASS = CALF
-----------------------
COCO has no dedicated "calf" class, so YOLO's "cow" class (id 19) is used
as a stand-in. A newborn calf is visually a small/young cow.

TWO FUNCTIONS ARE EXPOSED
--------------------------
1. detect_calf(frame)
   -> The original, simple contract. Returns the SINGLE best "cow"
      detection in the frame. This is what any external module should
      use if it only needs "is there a calf, and where".

2. detect_all_cows(frame)
   -> Returns EVERY "cow" detection in the frame, sorted by box area
      (smallest first). Useful when both a calf and its mother are in
      frame together: the smallest box is assumed to be the calf, and
      any larger box is assumed to be the mother. Used internally by
      the behaviour-detection module for the suckling heuristic.

INPUT (both functions)
-----------------------
    frame : numpy.ndarray - a single BGR frame (from cv2.VideoCapture
            or cv2.imread).

OUTPUT
------
    detect_calf(frame) ->
        {"detected": bool, "bbox": [x1,y1,x2,y2] or None, "confidence": float or None}

    detect_all_cows(frame) ->
        [{"bbox": [x1,y1,x2,y2], "confidence": float}, ...]   (possibly empty list)
"""

from ultralytics import YOLO
import numpy as np

# Loaded once at import time. Ultralytics auto-downloads yolov8n.pt on first use.
_model = YOLO("yolov8n.pt")

# COCO class id for "cow" (0-indexed, standard 80-class list).
COW_CLASS_ID = 19

# Minimum confidence for a detection to be trusted.
CONFIDENCE_THRESHOLD = 0.4


def _box_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def detect_all_cows(frame: np.ndarray) -> list:
    """Return every cow/calf-class detection in the frame, smallest box first."""
    if frame is None:
        return []

    results = _model(frame, verbose=False)
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == COW_CLASS_ID and conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = box.xyxy[0]
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            detections.append({"bbox": bbox, "confidence": round(conf, 2)})

    detections.sort(key=lambda d: _box_area(d["bbox"]))
    return detections


def detect_calf(frame: np.ndarray) -> dict:
    """
    Original simple contract: returns the highest-confidence single
    cow/calf detection in the frame.
    """
    cows = detect_all_cows(frame)
    if not cows:
        return {"detected": False, "bbox": None, "confidence": None}

    best = max(cows, key=lambda d: d["confidence"])
    return {"detected": True, "bbox": best["bbox"], "confidence": best["confidence"]}


# ---------------------------------------------------------------------------
# Self-test: python detection/calf_detection.py path/to/image_or_video
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import cv2

    if len(sys.argv) < 2:
        print("Usage: python calf_detection.py <path_to_image_or_video>")
        sys.exit(1)

    path = sys.argv[1]

    def draw(frame, result):
        if result["detected"]:
            x1, y1, x2, y2 = result["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Calf {result["confidence"]}', (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = detect_calf(frame)
            print(result)
            cv2.imshow("Calf Detection Test", draw(frame, result))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        frame = cv2.imread(path)
        result = detect_calf(frame)
        print(result)
        cv2.imshow("Calf Detection Test", draw(frame, result))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
