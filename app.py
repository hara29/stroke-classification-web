import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

MODEL_PATH = "best_ct_model.h5"
MODEL_URL = "https://huggingface.co/hara29/stroke-ct-model/resolve/main/best_ct_model.h5"
CLASS_NAMES = ['hemoragik', 'iskemik', 'normal']

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model dari Hugging Face..."):
            r = requests.get(MODEL_URL, stream=True)
            if r.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error("Gagal mengunduh model.")
                st.stop()

def preprocess_image(img):
    # konversi ke grayscale jika masih RGB
    if img.mode != 'L':
        img = img.convert('L')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)   
    return img_array

def predict(img):
    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    pred_conf = np.max(preds) * 100
    return pred_class, pred_conf

st.set_page_config(page_title="CT Scan Stroke Classification", layout="centered")
st.title("🧠 CT Scan Stroke Classification")

download_model()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload gambar CT Scan (.png/.jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300)

    if st.button("Prediksi"):
        pred_class, pred_conf = predict(img)
        st.success(f"**{pred_class}** — {pred_conf:.2f}%")
else:
    st.info("Silakan upload gambar untuk memulai.")
