import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Konfigurasi Halaman Web
st.set_page_config(page_title="Deteksi Kualitas Apel", page_icon="🍎")

st.title("🍎 Aplikasi Deteksi Kualitas Buah Apel")
st.write("Sistem Computer Vision untuk mengklasifikasi apel berdasarkan citra digital.")

# 1. Fungsi Load Model (Di-cache agar aplikasi tidak lambat)
@st.cache_resource
def load_model_keras():
    return tf.keras.models.load_model('model_kualitas_apel.h5')

try:
    model = load_model_keras()
    # Urutan kelas berdasarkan abjad dari ImageDataGenerator (Bagus=0, Bercak=1, Busuk=2)
    kelas_label = ['Bagus', 'Bercak', 'Busuk'] 
except:
    st.error("Model 'model_kualitas_apel.h5' tidak ditemukan. Pastikan file berada di folder yang sama.")
    st.stop()

# 2. Fitur Upload Gambar
uploaded_file = st.file_uploader("Unggah citra apel Anda di sini...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca dan menampilkan gambar asli
    image = Image.open(uploaded_file)
    st.image(image, caption="Citra Asli", width=300)
    
    # Konversi PIL Image ke format Array OpenCV
    img_array = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # --- BAGIAN MODIFIKASI PROSES (Untuk Demo Dosen) ---
    st.subheader("⚙️ Visualisasi Modifikasi Pengolahan Citra")
    col1, col2 = st.columns(2)
    
    # Transformasi HSV & Deteksi Tepi
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    
    with col1:
        # Menampilkan gambar HSV dengan mapping BGR agar warnanya terlihat terenkode
        st.image(img_hsv, caption="Transformasi HSV", channels="BGR", use_column_width=True)
    with col2:
        st.image(edges, caption="Canny Edge Detection", use_column_width=True)
        
    # --- BAGIAN PREDIKSI AI ---
    st.subheader("📊 Hasil Klasifikasi AI")
    
    # Preprocessing menyamakan dengan saat training
    img_resized = cv2.resize(img_array, (224, 224))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0) # Menambah dimensi batch
    
    # Eksekusi Prediksi
    prediksi = model.predict(img_input)
    indeks = np.argmax(prediksi)
    confidence = np.max(prediksi) * 100
    hasil = kelas_label[indeks]
    
    # Tampilan Hasil Interaktif
    if hasil == 'Bagus':
        st.success(f"**Kualitas Terdeteksi:** {hasil} (Confidence: {confidence:.2f}%)")
    elif hasil == 'Bercak':
        st.warning(f"**Kualitas Terdeteksi:** {hasil} (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"**Kualitas Terdeteksi:** {hasil} (Confidence: {confidence:.2f}%)")