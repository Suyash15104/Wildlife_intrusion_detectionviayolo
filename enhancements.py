import cv2
from collections import deque

# ================= TEMPORAL MEMORY =================
FRAME_HISTORY = 5
history = deque(maxlen=FRAME_HISTORY)

# ================= CLAHE =================
def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ================= FILTERING =================
def filter_boxes(results, conf_threshold=0.4, min_area=500):
    detections = {"human": False, "tiger": False, "elephant": False}

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)

            if conf < conf_threshold or area < min_area:
                continue

            if cls == 0:
                detections["tiger"] = True
            elif cls == 1:
                detections["human"] = True
            elif cls == 2:
                detections["elephant"] = True

    return detections


# ================= TEMPORAL CONSISTENCY =================
def update_history(detection):
    history.append(detection)

def check_consistency(min_frames=3):
    count = {"human": 0, "tiger": 0, "elephant": 0}

    for h in history:
        for key in h:
            if h[key]:
                count[key] += 1

    return {k: count[k] >= min_frames for k in count}