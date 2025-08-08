import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import base64
from io import BytesIO

# ======================
# Konfigurasi
# ======================
MODEL_PATH = "best_ct_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1_ckKQ2PFAhLJ4lSKK_bc8wIPEO_r72Ou"
CLASS_NAMES = ['hemoragik', 'iskemik', 'normal']

# ======================
# Fungsi Download Model
# ======================
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model..."):
            try:
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Gagal mengunduh model: {e}")
                st.stop()

# ======================
# Preprocessing
# ======================
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ======================
# Prediksi
# ======================
def predict(img):
    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    pred_conf = np.max(preds) * 100
    return pred_class, pred_conf

# ======================
# Fungsi Preview Center
# ======================
def show_image_center(img, max_width=300):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{img_b64}" 
                 style="max-width:{max_width}px; border-radius:10px;"/>
        </div>
        """,
        unsafe_allow_html=True
    )

# ======================
# UI Streamlit
# ======================
st.set_page_config(page_title="CT Scan Stroke Classification", layout="centered")
st.title("🧠 CT Scan Stroke Classification")

# Download model jika belum ada
download_model()

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload gambar CT Scan (.png/.jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    show_image_center(img, max_width=300)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        if st.button("Prediksi"):
            pred_class, pred_conf = predict(img)
            st.success(f"**{pred_class}** — {pred_conf:.2f}%")
    with col_btn3:
        if st.button("Hapus Gambar"):
            st.experimental_rerun()
else:
    st.info("Silakan upload gambar untuk memulai.")
