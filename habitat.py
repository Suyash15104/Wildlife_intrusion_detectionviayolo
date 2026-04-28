import streamlit as st
import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
from ultralytics import YOLO

# Load once
@st.cache_resource
def load_models():
    return YOLO("yolov8n.pt")

# COCO class IDs for habitat
HABITAT_CLASSES = [1, 2, 3, 7]  # bicycle, car, motorcycle, truck


def detect_habitat(frame):
    model_habitat = load_models()
    results = model_habitat(frame)

    habitat_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls in HABITAT_CLASSES:
                habitat_count += 1

    habitat = habitat_count >= 1

    return habitat, habitat_count