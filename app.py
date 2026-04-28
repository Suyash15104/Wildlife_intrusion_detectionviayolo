import streamlit as st
import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from ultralytics import YOLO
import cv2
import tempfile
import time
import smtplib

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
    return YOLO(MODEL_PATH)

# ================= EMAIL =================
def send_email_alert(detected_objects, image_path):
    sender   = st.secrets["EMAIL_USER"]
    password = st.secrets["EMAIL_PASS"]
    receiver = st.secrets["EMAIL_RECEIVER"]

    msg = MIMEMultipart()
    msg["From"]    = sender
    msg["To"]      = receiver
    msg["Subject"] = "🚨 Intrusion Alert"

    body = f"Intrusion detected: {', '.join(detected_objects)}"
    msg.attach(MIMEText(body, "plain"))

    with open(image_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())

    encoders.encode_base64(part)
    part.add_header("Content-Disposition",
                    f"attachment; filename={os.path.basename(image_path)}")
    msg.attach(part)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender, password)
    server.send_message(msg)
    server.quit()


# ================= PROCESS FRAME =================
def process_frame(frame, conf, use_clahe):
    if use_clahe:
        frame = apply_clahe(frame)

    model = load_model()
    results = model.predict(frame, conf=conf)

    detections = filter_boxes(results)
    update_history(detections)
    consistent = check_consistency()

    human    = consistent["human"]
    tiger    = consistent["tiger"]
    elephant = consistent["elephant"]

    detected = []
    if human:    detected.append("Human")
    if tiger:    detected.append("Tiger")
    if elephant: detected.append("Elephant")

    habitat, habitat_count = detect_habitat(frame)

    annotated = results[0].plot()

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated, ts, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if human and (tiger or elephant):
        status = "🚨 Human-Wildlife Conflict"

        if not st.session_state.intrusion_active:
            st.session_state.intrusion_count += 1

            if time.time() - st.session_state.last_alert_time > 10:
                if not os.path.exists("alerts"):
                    os.makedirs("alerts")

                path = f"alerts/intrusion_{int(time.time())}.jpg"
                cv2.imwrite(path, annotated)

                try:
                    send_email_alert(detected, path)
                except Exception as e:
                    st.warning(f"Email alert failed: {e}")

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
mode     = st.sidebar.selectbox("Input Source", ["Upload Video", "Live Camera"])
conf     = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)
use_clahe = st.sidebar.checkbox("Low-Light Enhancement", True)

col_start, col_stop = st.sidebar.columns(2)
start = col_start.button("▶ Start")
stop  = col_stop.button("⏹ Stop")

# ================= STATE =================
for key, default in [
    ("intrusion_count", 0),
    ("last_alert_time", 0),
    ("intrusion_active", False),
    ("running", False),
    ("cap_path", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# ================= FIXED UI =================
col1, col2, col3 = st.columns(3)
metric_h    = col1.empty()
metric_t    = col2.empty()
metric_e    = col3.empty()
status_box  = st.empty()
frame_box   = st.empty()
fps_box     = st.sidebar.empty()
habitat_box = st.sidebar.empty()

# ================= ALERT HISTORY =================
st.sidebar.subheader("📁 Alert History")
if os.path.exists("alerts"):
    files = sorted(os.listdir("alerts"))[-5:]
    for f in files:
        st.sidebar.text(f)
st.sidebar.metric("Total Intrusions", st.session_state.intrusion_count)


# ================= VIDEO =================
if mode == "Upload Video":
    file = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if file and st.session_state.running:
        # Write upload to temp file once
        if st.session_state.cap_path is None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(file.read())
            tmp.flush()
            st.session_state.cap_path = tmp.name

        cap = cv2.VideoCapture(st.session_state.cap_path)

        # Process ONE frame per Streamlit rerun — avoids blocking the event loop
        if cap.isOpened():
            start_t = time.time()
            ret, frame = cap.read()

            if ret and st.session_state.running:
                annotated, status, h, t, e, hc = process_frame(frame, conf, use_clahe)

                metric_h.metric("Humans", int(h))
                metric_t.metric("Wildlife A", int(t))
                metric_e.metric("Wildlife B", int(e))
                habitat_box.metric("Habitat Objects", hc)

                color = "red" if "🚨" in status else "orange" if "⚠" in status else "green"
                status_box.markdown(
                    f"<div style='background:{color};padding:10px;border-radius:10px;"
                    f"text-align:center'><h3>{status}</h3></div>",
                    unsafe_allow_html=True,
                )
                frame_box.image(annotated, channels="BGR")

                fps = int(1 / max(time.time() - start_t, 1e-6))
                fps_box.write(f"FPS: {fps}")

                # Trigger next rerun after a short delay
                time.sleep(0.03)
                st.rerun()
            else:
                st.success("✅ Video processing complete.")
                st.session_state.running = False
                st.session_state.cap_path = None

        cap.release()

    elif file and not st.session_state.running:
        # Reset cap path when stopped so re-upload works
        st.session_state.cap_path = None


# ================= CAMERA =================
elif mode == "Live Camera":
    st.info("⚠️ Live camera requires a local deployment. It is not supported on Streamlit Cloud.")