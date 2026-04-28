import streamlit as st
import os
os.environ["YOLO_CONFIG_DIR"]="/tmp/Ultralytics"
from ultralytics import YOLO
import cv2
import tempfile
import time
import smtplib
import os

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# ===== IMPORT MODULES =====
from enhancements import apply_clahe, filter_boxes, update_history, check_consistency
from habitat import detect_habitat

# ================= CONFIG =================
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO("best.pt")
# ================= EMAIL =================
def send_email_alert(detected_objects, image_path):
    sender = st.secrets["EMAIL_USER"]
    password = st.secrets["EMAIL_PASS"]
    receiver = st.secrets["EMAIL_RECEIVER"]

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = "🚨 Intrusion Alert"

    body = f"Intrusion detected: {', '.join(detected_objects)}"
    msg.attach(MIMEText(body, "plain"))

    with open(image_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())

    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
    msg.attach(part)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender, password)
    server.send_message(msg)
    server.quit()


# ================= MODEL =================

# ================= UI =================
st.set_page_config(layout="wide")

st.markdown("""
<style>
.stApp {background-color:#0E1117; color:white;}
</style>
""", unsafe_allow_html=True)

st.title("🌿 Wildlife Intrusion Detection System")
st.markdown("### Context-Aware Multi-Species Monitoring Dashboard")

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Controls")

mode = st.sidebar.selectbox("Input Source", ["Upload Video", "Live Camera"])
conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)
use_clahe = st.sidebar.checkbox("Low-Light Enhancement", True)

start = st.sidebar.button("▶ Start")
stop = st.sidebar.button("⏹ Stop")

# ================= STATE =================
if "intrusion_count" not in st.session_state:
    st.session_state.intrusion_count = 0

if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0

if "intrusion_active" not in st.session_state:
    st.session_state.intrusion_active = False


# ================= FIXED UI =================
top = st.container()

with top:
    col1, col2, col3 = st.columns(3)
    metric_h = col1.empty()
    metric_t = col2.empty()
    metric_e = col3.empty()

    status_box = st.empty()
    frame_box = st.empty()

fps_box = st.sidebar.empty()
habitat_box = st.sidebar.empty()

# ================= ALERT HISTORY =================
st.sidebar.subheader("📁 Alert History")
if os.path.exists("alerts"):
    for f in os.listdir("alerts")[-5:]:
        st.sidebar.text(f)

st.sidebar.metric("Total Intrusions", st.session_state.intrusion_count)


# ================= PROCESS FRAME =================
def process_frame(frame):

    if use_clahe:
        frame = apply_clahe(frame)

    # ===== MAIN MODEL =====
    model = load_model()
    results = model.predict(frame)

    detections = filter_boxes(results)
    update_history(detections)
    consistent = check_consistency()

    human = consistent["human"]
    tiger = consistent["tiger"]
    elephant = consistent["elephant"]

    detected = []
    if human: detected.append("Human")
    if tiger: detected.append("Tiger")
    if elephant: detected.append("Elephant")

    # ===== HABITAT DETECTION =====
    habitat, habitat_count = detect_habitat(frame)

    annotated = results[0].plot()

    # timestamp overlay
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated, ts, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

    # ===== CONTEXT-AWARE LOGIC =====
    if human and (tiger or elephant):
        status = f"🚨 Human-Wildlife Conflict"

        if not st.session_state.intrusion_active:
            st.session_state.intrusion_count += 1

            if time.time() - st.session_state.last_alert_time > 10:

                if not os.path.exists("alerts"):
                    os.makedirs("alerts")

                path = f"alerts/intrusion_{int(time.time())}.jpg"
                cv2.imwrite(path, annotated)

                send_email_alert(detected, path)

                st.session_state.last_alert_time = time.time()

        st.session_state.intrusion_active = True

    elif (tiger or elephant) and habitat:
        status = f"⚠ Wildlife in Human Habitat ({habitat_count} vehicle(s))"
        st.session_state.intrusion_active = False

    elif tiger or elephant:
        status = "🐾 Wildlife in Natural Area"
        st.session_state.intrusion_active = False

    elif human:
        status = "👤 Human Detected"
        st.session_state.intrusion_active = False

    else:
        status = "✅ Normal"
        st.session_state.intrusion_active = False

    return annotated, status, human, tiger, elephant, habitat_count


# ================= VIDEO =================
if mode == "Upload Video":

    file = st.file_uploader("Upload Video", type=["mp4","avi"])

    if file and start:

        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())

        cap = cv2.VideoCapture(temp.name)

        while cap.isOpened():

            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            annotated, status, h, t, e, habitat_count = process_frame(frame)

            # metrics
            metric_h.metric("Humans", int(h))
            metric_t.metric("Wildlife A", int(t))
            metric_e.metric("Wildlife B", int(e))

            habitat_box.metric("Habitat Objects", habitat_count)

            # status banner
            color = "red" if "🚨" in status else "orange" if "⚠" in status else "green"
            status_box.markdown(
                f"<div style='background:{color};padding:10px;border-radius:10px;text-align:center'><h3>{status}</h3></div>",
                unsafe_allow_html=True
            )

            frame_box.image(annotated, channels="BGR")

            fps = int(1/(time.time()-start_time))
            fps_box.write(f"FPS: {fps}")

            if stop:
                break

            time.sleep(0.01)

        cap.release()


# ================= CAMERA =================
elif mode == "Live Camera":

    if start:

        cap = cv2.VideoCapture(0)

        while True:

            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            annotated, status, h, t, e, habitat_count = process_frame(frame)

            metric_h.metric("Humans", int(h))
            metric_t.metric("Wildlife A", int(t))
            metric_e.metric("Wildlife B", int(e))

            habitat_box.metric("Habitat Objects", habitat_count)

            color = "red" if "🚨" in status else "orange" if "⚠" in status else "green"
            status_box.markdown(
                f"<div style='background:{color};padding:10px;border-radius:10px;text-align:center'><h3>{status}</h3></div>",
                unsafe_allow_html=True
            )

            frame_box.image(annotated, channels="BGR")

            fps = int(1/(time.time()-start_time))
            fps_box.write(f"FPS: {fps}")

            if stop:
                break

            time.sleep(0.01)

        cap.release()