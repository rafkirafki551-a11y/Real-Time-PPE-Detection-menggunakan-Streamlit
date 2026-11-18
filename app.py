import os
import av
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="PPE Detection System", page_icon="🦺", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    return YOLO(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


class PPEProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence = 0.35

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(
            source=img,
            conf=self.confidence,
            imgsz=416,
            device="cpu",
            verbose=False,
        )
        return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")


st.title("🦺 PPE Detection System")
st.markdown("### Real-Time Personal Protective Equipment Detection")
st.divider()

st.sidebar.header("Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, 0.35, 0.05)
st.sidebar.success("Model berhasil dimuat")
st.sidebar.write("**Kelas terdeteksi:**")
for name in model.names.values():
    st.sidebar.write(f"- {name}")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Live Camera Detection")
    ctx = webrtc_streamer(
        key="ppe-detection",
        video_processor_factory=PPEProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    # Kirim nilai slider ke processor yang sedang berjalan
    if ctx.video_processor:
        ctx.video_processor.confidence = confidence

with col2:
    st.subheader("Model Information")
    st.metric("Detection Model", "YOLO")
    st.metric("Confidence", f"{confidence:.0%}")
    st.info("Arahkan kamera ke objek untuk melakukan deteksi APD.")

st.divider()
st.caption("PPE Detection System | Computer Vision | YOLO")
