import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Malaria Cell Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Custom CSS ─────────────────────────────────────────────────────────
with open("style.html", "r") as f:
    html_css = f.read()
st.markdown(html_css, unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Trophozoite", "WBC", "NEG"]
CLASS_DESCRIPTIONS = {
    "Trophozoite": "🦠 **Malaria Parasite Detected** — Trophozoite is an early-stage malaria parasite inside a red blood cell. Immediate medical attention advised.",
    "WBC": "🩸 **White Blood Cell Detected** — No malaria parasite found. This appears to be a normal white blood cell.",
    "NEG": "✅ **Negative / No Parasite** — No malaria-related cells detected in this region.",
}
CLASS_COLORS = {
    "Trophozoite": "#e74c3c",
    "WBC": "#3498db",
    "NEG": "#2ecc71",
}
MODEL_PATH = "malaria_hybrid_model.keras"

# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file `{MODEL_PATH}` not found. Please place it in the same directory as `app.py`.")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ─── Helper Functions ────────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize and normalise a PIL image for the CNN branch."""
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE)
    return img.astype(np.float32) / 255.0

def extract_spatial_features(pil_image: Image.Image,
                               xmin=None, ymin=None,
                               xmax=None, ymax=None) -> np.ndarray:
    """
    Compute the 5 spatial features the model was trained on.
    If the user provides bounding-box coordinates use them;
    otherwise fall back to using the whole image as the bounding box.
    """
    w_full, h_full = pil_image.size

    xmin = float(xmin) if xmin is not None else 0.0
    ymin = float(ymin) if ymin is not None else 0.0
    xmax = float(xmax) if xmax is not None else float(w_full)
    ymax = float(ymax) if ymax is not None else float(h_full)

    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    return np.array([[x_center, y_center, width, height, area]], dtype=np.float32)

def predict(pil_image, spatial_feats):
    img_array = preprocess_image(pil_image)
    img_batch = np.expand_dims(img_array, axis=0)               # (1, 224, 224, 3)

    raw_preds = model.predict([img_batch, spatial_feats], verbose=0)[0]
    predicted_idx = int(np.argmax(raw_preds))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(raw_preds[predicted_idx])
    return predicted_class, confidence, raw_preds

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Malaria_lifecycle.jpg/640px-Malaria_lifecycle.jpg",
             caption="Malaria parasite lifecycle", use_container_width=True)
    st.markdown("---")
    st.markdown("### ⚙️ Advanced: Bounding Box")
    st.markdown("Optionally supply the cell's pixel coordinates for more accurate spatial features.")
    use_bbox = st.checkbox("Provide bounding box coordinates", value=False)
    xmin = ymin = xmax = ymax = None
    if use_bbox:
        col1, col2 = st.columns(2)
        with col1:
            xmin = st.number_input("xmin", min_value=0, value=0)
            ymin = st.number_input("ymin", min_value=0, value=0)
        with col2:
            xmax = st.number_input("xmax", min_value=1, value=224)
            ymax = st.number_input("ymax", min_value=1, value=224)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "This app uses a **hybrid deep-learning model** that fuses:\n"
        "- 🖼️ **EfficientNetB0** visual features\n"
        "- 📐 **Spatial metadata** (centroid, area)\n\n"
        "Classes: `Trophozoite` · `WBC` · `NEG`"
    )

# ─── Main UI ─────────────────────────────────────────────────────────────────
st.markdown('<div class="header-block">', unsafe_allow_html=True)
st.title("🔬 Malaria Cell Classifier")
st.markdown(
    "Upload a **blood smear cell image** to detect whether it contains a malaria parasite (Trophozoite), "
    "a white blood cell (WBC), or is negative (NEG)."
)
st.markdown("</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a cell image here (JPG / PNG / BMP)",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is not None:
    pil_image = Image.open(io.BytesIO(uploaded_file.read()))

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### Uploaded Image")
        st.image(pil_image, use_container_width=True, caption=uploaded_file.name)

    with col_result:
        st.markdown("#### Prediction")
        with st.spinner("Analysing cell…"):
            spatial = extract_spatial_features(pil_image, xmin, ymin, xmax, ymax)
            pred_class, confidence, raw_preds = predict(pil_image, spatial)

        color = CLASS_COLORS[pred_class]
        st.markdown(
            f'<div class="result-card" style="border-left: 6px solid {color};">'
            f'<h2 style="color:{color};">{pred_class}</h2>'
            f'<p style="font-size:1.1rem;">Confidence: <strong>{confidence*100:.1f}%</strong></p>'
            f'<p>{CLASS_DESCRIPTIONS[pred_class]}</p>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Class Probabilities")
        for i, cls in enumerate(CLASS_NAMES):
            prob = float(raw_preds[i])
            st.markdown(f"**{cls}**")
            st.progress(prob, text=f"{prob*100:.1f}%")

    # ─── Disclaimer ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer:** This tool is for research and educational purposes only. "
        "It is **not** a certified medical diagnostic device. Always consult a qualified "
        "healthcare professional for clinical diagnosis."
    )

else:
    st.info("👆 Upload an image to get started.")

    st.markdown("---")
    st.markdown("### 🧬 How It Works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Upload**\n\nDrop any blood smear cell crop (JPG / PNG).")
    with c2:
        st.markdown("**2. Hybrid Analysis**\n\nEfficientNetB0 extracts visual features; spatial metadata boosts accuracy.")
    with c3:
        st.markdown("**3. Result**\n\nGet class prediction + per-class confidence scores instantly.")
