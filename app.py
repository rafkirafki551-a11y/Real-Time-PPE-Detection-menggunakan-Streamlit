
import streamlit as st
import cv2
import av
import os

from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

# =====================================
# 1. CONFIGURATION
# =====================================

st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🦺",
    layout="wide"
)

MODEL_PATH = "model.pt"

# =====================================
# 2. LOAD MODEL
# =====================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    return YOLO(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# =====================================
# 3. VIDEO PROCESSOR
# =====================================

class PPEProcessor(VideoProcessorBase):

    def __init__(self):
        self.confidence = 0.35

    def recv(self, frame):

        # Convert frame to OpenCV
        img = frame.to_ndarray(
            format="bgr24"
        )

        # YOLO Prediction
        results = model.predict(
            source=img,
            conf=self.confidence,
            verbose=False
        )

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


# =====================================
# 4. USER INTERFACE
# =====================================

st.title("🦺 PPE Detection System")

st.markdown(
    """
    ### Real-Time Personal Protective Equipment Detection

    Sistem deteksi penggunaan Alat Pelindung Diri
    menggunakan Deep Learning dan YOLO.
    """
)

st.divider()

# Sidebar
st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05
)

st.sidebar.markdown("---")

st.sidebar.success("Model berhasil dimuat")

st.sidebar.write(
    "Model: YOLO"
)

# Main layout
col1, col2 = st.columns([3, 1])

with col1:

    st.subheader("Live Camera Detection")

    ctx = webrtc_streamer(
        key="ppe-detection",
        video_processor_factory=PPEProcessor,

        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )

with col2:

    st.subheader("Model Information")

    st.metric(
        "Detection Model",
        "YOLO"
    )

    st.metric(
        "Confidence",
        f"{confidence:.0%}"
    )

    st.info(
        "Arahkan kamera ke objek "
        "untuk melakukan deteksi APD."
    )

st.divider()

st.caption(
    "PPE Detection System | Computer Vision | YOLO"
)
