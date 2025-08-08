import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ======================
# Load model
# ======================
MODEL_PATH = "best_ct_model.h5"
MODEL_URL ="https://drive.google.com/uc?id=1_ckKQ2PFAhLJ4lSKK_bc8wIPEO_r72Ou"

# Kelas sesuai dataset
CLASS_NAMES = ['hemoragik', 'iskemik', 'normal']

# ===== FUNGSI DOWNLOAD MODEL =====
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model..."):
            r = requests.get(MODEL_URL, stream=True)
            if r.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error("Gagal mengunduh model.")
                st.stop()

# ===== PREPROCESSING =====
def preprocess_image(img):
    img = img.resize((224, 224))  # Sesuaikan ukuran input model
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
model = tf.keras.models.load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload gambar CT Scan (.png/.jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Tampilkan preview kecil di tengah
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{st.image(img, use_container_width=False, output_format='PNG').data}" 
                 style="max-width:300px; border-radius:10px;"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Tombol aksi di tengah
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn1:
        if st.button("Prediksi"):
            pred_class, pred_conf = predict(img)
            st.success(f"**{pred_class}** — {pred_conf:.2f}%")
    with col_btn2:
        if st.button("Hapus Gambar"):
            uploaded_file = None
            st.experimental_rerun()
else:
    st.info("Silakan upload gambar untuk memulai.")
