import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import pandas as pd
import io

st.set_page_config(page_title="Tomato Leaf Disease Detector", layout="centered")
st.title("🍅 Tomato Leaf Disease Detection")

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return YOLO(path)

model = load_model(MODEL_PATH)
if model is None:
    st.error(f"Model not found at '{MODEL_PATH}'. Put your best.pt in the app folder.")
    st.stop()

st.write("### Upload a tomato leaf image (jpg, png).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
conf_thresh = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
imgsz = st.sidebar.selectbox("Inference image size", [320], index=0)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running YOLO inference..."):
        results = model.predict(source=np.array(image), conf=conf_thresh, imgsz=imgsz)
    r = results[0]

    # Annotated image
    annotated = r.plot()
    st.image(annotated, caption="Detection Output", use_container_width=True)

    # Save annotated image for download
    annotated_pil = Image.fromarray(annotated[..., ::-1])  # Convert BGR->RGB
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="📥 Download Annotated Image",
        data=byte_im,
        file_name="annotated_result.png",
        mime="image/png"
    )

    # Show detections
    st.subheader("Prediction Results")
    cls_tensor = getattr(r.boxes, "cls", None)
    conf_tensor = getattr(r.boxes, "conf", None)
    if cls_tensor is None or len(cls_tensor) == 0:
        st.write("No detections above the threshold.")
    else:
        # Convert to DataFrame
        detections = [
            {"Class": model.names[int(cls_id)], "Confidence": float(conf)}
            for cls_id, conf in zip(cls_tensor, conf_tensor)
        ]
        df = pd.DataFrame(detections)
        st.table(df)

        # Download predictions as CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
