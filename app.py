import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown  # pastikan ada di requirements.txt

# ======================
# Load model
# ======================
MODEL_PATH = "best_ct_model.h5"
MODEL_ID = "1_ckKQ2PFAhLJ4lSKK_bc8wIPEO_r72Ou"  # ID dari Google Drive
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

# Kelas sesuai dataset
CLASS_NAMES = ['hemoragik', 'iskemik', 'normal']

# ===== FUNGSI DOWNLOAD MODEL =====
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model dari Google Drive..."):
            try:
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Gagal mengunduh model: {e}")
                st.stop()
        if not os.path.exists(MODEL_PATH):
            st.error("Model gagal diunduh atau file corrupt.")
            st.stop()

# ===== PREPROCESSING =====
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ===== PREDIKSI =====
def predict(img):
    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    pred_conf = np.max(preds) * 100
    return pred_class, pred_conf

# ===== UI STREAMLIT =====
st.set_page_config(page_title="CT Scan Stroke Classification", layout="centered")
st.title("🧠 CT Scan Stroke Classification")

# Download model jika belum ada
download_model()

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload gambar CT Scan (.png/.jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Preview gambar lebih kecil & center
    st.markdown(
        "<div style='text-align:center;'>",
        unsafe_allow_html=True
    )
    st.image(img, width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Prediksi"):
            pred_class, pred_conf = predict(img)
            st.success(f"**{pred_class}** — {pred_conf:.2f}%")
    with col2:
        if st.button("Hapus Gambar"):
            st.experimental_rerun()
else:
    st.info("Silakan upload gambar untuk memulai.")
