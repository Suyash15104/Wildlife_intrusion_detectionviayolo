# Wildlife_intrusion_detectionviayolo
# Context-Aware Wildlife Intrusion Detection System using YOLOv8

A real-time multi-species wildlife intrusion detection system built with custom-trained YOLOv8 models. Detects Tiger, Human, Elephant, and Bear from video feeds and classifies intrusion scenarios using context-aware decision logic. Deployed as an interactive Streamlit dashboard with automated email alerts.

---

## Project Structure

```
wildlife-intrusion-detection/

 app.py # Main Streamlit application
 enhancements.py # CLAHE preprocessing, bounding box filtering, temporal consistency
 habitat.py # Pretrained YOLOv8 habitat detection (vehicles)
 requirements.txt # Python dependencies
 README.md # This file

 alerts/ # Auto-created: stores intrusion screenshots

 runs/
 detect/
 train10/
 weights/
 best.pt # Your custom-trained YOLOv8 model weights
```

---

## How It Works

The system processes each video frame through 8 sequential layers:

```
Video Input
 ↓
CLAHE Preprocessing (low-light enhancement)
 ↓
Custom YOLOv8 Detection (Tiger, Human, Elephant, Bear)
 ↓
Habitat YOLOv8 Detection (bicycle, car, motorcycle, truck)
 ↓
Bounding Box Filtering (confidence + area threshold)
 ↓
Temporal Consistency (deque sliding window validation)
 ↓
Decision Engine (3-rule context classification)
 ↓
Streamlit Dashboard + Email Alert
```

**Classification Rules:**

| Scene Condition | Status |
|---|---|
| Wildlife + Human detected | Human-Wildlife Conflict |
| Wildlife + Vehicle detected | Wildlife in Human Habitat |
| Wildlife only | Wildlife in Natural Area |
| Human only | Human Detected |
| Nothing detected | Normal |

---

## Requirements

- Python 3.9 or higher
- NVIDIA GPU recommended (CUDA 11.8+)
- Windows 10/11 or Ubuntu 20.04+

---

## Installation

### Step 1 — Clone or download the project

```bash
git clone https://github.com/yourusername/wildlife-intrusion-detection.git
cd wildlife-intrusion-detection
```

Or simply download and extract the ZIP folder.

---

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt`, install manually:

```bash
pip install ultralytics streamlit opencv-python numpy pillow
```

---

### Step 4 — Set up the model weights

Place your custom-trained model weights file (`best.pt`) in the correct location.

Open `app.py` and update the `MODEL_PATH` variable to point to your weights file:

```python
MODEL_PATH = r"C:\Users\YourName\runs\detect\train10\weights\best.pt"
```

> **Note:** The model was trained on 4 classes: Tiger (0), Human (1), Elephant (2), Bear (3)

---

### Step 5 — Configure email alerts

The system sends automatic email alerts on intrusion detection. Update the email settings in `app.py`:

```python
sender = "your_gmail@gmail.com"
password = "your_app_password" # Gmail App Password, not your regular password
receiver = "alert_recipient@gmail.com"
```

**To generate a Gmail App Password:**
1. Go to your Google Account → Security
2. Enable 2-Step Verification if not already enabled
3. Go to Security → App Passwords
4. Select app: Mail, device: Windows Computer
5. Copy the 16-character password and paste it as `password`

> **Security:** Never share your app password publicly. Do not push it to GitHub.

---

## Running the Application

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at:
```
http://localhost:8501
```

---

## Using the Dashboard

Once the app opens in your browser:

**Sidebar controls:**

| Control | Description |
|---|---|
| Input Source | Choose between Upload Video or Live Camera |
| Confidence slider | Detection confidence threshold (recommended: 0.15 to 0.25) |
| Low-Light Enhancement | Toggle CLAHE preprocessing for dark footage |
| Start | Begin processing |
| ⏹ Stop | Stop processing |

**Steps to run on a video file:**
1. Select **Upload Video** from the Input Source dropdown
2. Click **Browse files** and select your `.mp4` or `.avi` video
3. Adjust the Confidence slider (start with 0.15 for better detection)
4. Enable **Low-Light Enhancement** if your video is dark
5. Click ** Start**

**Steps to run on live camera:**
1. Select **Live Camera** from the Input Source dropdown
2. Make sure your webcam is connected
3. Click ** Start**

---

## Dashboard Features

| Feature | Description |
|---|---|
| Live video display | Annotated frames with bounding boxes streamed in real time |
| Humans counter | Count of human detections in current window |
| Wildlife A counter | Tiger detections |
| Wildlife B counter | Elephant detections |
| Habitat Objects | Count of vehicles detected by habitat model |
| Status banner | Colour-coded: Red (Intrusion), Orange (Wildlife in Zone), Green (Normal) |
| FPS display | Real-time processing speed |
| Alert History | Last 5 saved intrusion screenshots listed in sidebar |
| Total Intrusions | Running count of intrusion events in current session |

---

## Intrusion Alerts

When an intrusion is detected (Human + Wildlife in same frame):

- An annotated screenshot is saved automatically to the `alerts/` folder
- An email is sent to the configured recipient with the screenshot attached
- Alerts are throttled to one per 10 seconds to avoid spam
- The alert history panel in the sidebar lists recent alert filenames

---

## Detection Classes

| Class ID | Species | Bounding Box Colour |
|---|---|---|
| 0 | Tiger | Varies (YOLOv8 default palette) |
| 1 | Human | Varies |
| 2 | Elephant | Varies |
| 3 | Bear | Varies |

**Habitat detection classes** (pretrained COCO model):

| Class | Indication |
|---|---|
| Bicycle | Human pathway nearby |
| Car | Motorised habitation zone |
| Motorcycle | Rural village transport |
| Truck | Agricultural or transport zone |

---

## Troubleshooting

**Tiger not being detected in video:**
- Lower the Confidence slider to 0.10 or 0.15
- Enable Low-Light Enhancement even for daytime footage
- Make sure your `best.pt` path in `app.py` is correct
- Try a shorter, clearer video clip first to confirm the model loads correctly

**App crashes on start:**
- Check that `best.pt` exists at the path specified in `MODEL_PATH`
- Make sure all packages are installed: `pip install ultralytics streamlit opencv-python`
- Check that you are running from the correct folder

**Email alert not sending:**
- Confirm your Gmail App Password is correct (not your regular Gmail password)
- Make sure 2-Step Verification is enabled on your Google account
- Check that your internet connection is active during processing

**Low FPS (below 5):**
- Close other applications to free GPU memory
- Reduce the video resolution before uploading
- Use a smaller input: in `app.py` change `imgsz=640` to `imgsz=416`

**`alerts/` folder not created:**
- This is created automatically on the first intrusion detection
- Make sure the app has write permission in the project folder

---

## Retraining the Model

If you want to retrain on your own dataset:

```bash
yolo detect train \
 model=yolov8n.pt \
 data=dataset.yaml \
 epochs=100 \
 imgsz=640 \
 batch=4 \
 workers=0 \
 project=runs/detect \
 name=train_custom
```

Your `dataset.yaml` should look like:

```yaml
path: C:/Users/YourName/dataset
train: train/images
val: valid/images

nc: 4
names: ['tiger', 'human', 'elephant', 'bear']
```

After training, update `MODEL_PATH` in `app.py` to point to the new `best.pt`.

---

## Dataset Details

| Property | Details |
|---|---|
| Classes | Tiger (0), Human (1), Elephant (2), Bear (3) |
| Annotation format | YOLO normalised (class_id x_center y_center width height) |
| Annotation tool | LabelImg |
| Training images | ~192 (Phase 1), expanded in Phases 2 and 3 |
| Validation images | 40 (Phase 1), 64 (Phase 2) |

---

## ‍ Authors

| Name | Enrollment No. |
|---|---|
| Suyash Bagale | BT22ECE004 |
| Srisailam A | BT22ECE052 |

**Supervisor:** Dr. V. R. Satpute, Associate Professor, Department of Electronics and Communication Engineering, VNIT Nagpur

---

## Institution

Department of Electronics and Communication Engineering
Visvesvaraya National Institute of Technology, Nagpur — 440010 (India)

---

## License

This project was developed for academic purposes as part of the B.Tech final year project at VNIT Nagpur, 2026.
