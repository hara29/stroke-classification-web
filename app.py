import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ======================
# Load model
# ======================
MODEL_PATH = "best_ct_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Kelas sesuai dataset
class_names = ['hemoragik', 'iskemik', 'normal']

# ======================
# Preprocessing
# ======================
def preprocess_image(image):
    img = image.convert('L')  # Grayscale
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="CT Scan Stroke Classification", layout="centered")
st.title("🧠 CT Scan Stroke Classification")
st.write("Upload gambar CT scan (.png / .jpg) untuk deteksi **Hemorrhagic**, **Ischemic**, atau **Normal**.")

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# Upload gambar
uploaded_file = st.file_uploader(
    "Upload Gambar CT Scan",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    st.session_state.uploaded_image = uploaded_file

# Tombol hapus
if st.button("Hapus Gambar"):
    st.session_state.uploaded_image = None

# Preview + prediksi
if st.session_state.uploaded_image is not None:
    image = Image.open(st.session_state.uploaded_image)
    max_width = 300
    st.image(image, caption="CT Scan yang diupload", use_column_width=False, width=max_width)

    if st.button("Prediksi"):
        x = preprocess_image(image)
        preds = model.predict(x)
        pred_idx = np.argmax(preds[0])
        pred_label = class_names[pred_idx]
        pred_conf = preds[0][pred_idx] * 100

        st.subheader("Hasil Prediksi")
        st.write(f"**{pred_label.upper()}** ({pred_conf:.2f}%)")

        st.subheader("Probabilitas")
        prob_dict = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
        st.bar_chart(prob_dict)
else:
    st.info("Silakan upload gambar terlebih dahulu.")
