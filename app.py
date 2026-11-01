import cv2, joblib
import numpy as np
import streamlit as st
from src.pipeline import ekstrak_fitur

st.set_page_config(page_title="Deteksi Kualitas Buah", layout="wide")
st.title("Aplikasi Deteksi Kualitas Buah")

model = joblib.load("models/model_svm.pkl")

file = st.file_uploader("Upload foto buah", type=["jpg", "jpeg", "png"])
if file:
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    fitur, hasil, mask, cacat = ekstrak_fitur(img)
    proba = model.predict_proba([fitur])[0]
    idx = proba.argmax()
    label = model.classes_[idx]

    # grade sederhana dari rasio cacat (ubah ambangnya sesuai hasil percobaan)
    grade = "A" if cacat < 0.05 else "B" if cacat < 0.20 else "C"

    k1, k2, k3 = st.columns(3)
    k1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Citra asli")
    k2.image(mask, caption="Hasil segmentasi")
    k3.image(cv2.cvtColor(hasil, cv2.COLOR_BGR2RGB), caption="Buah tersegmentasi")

    st.subheader(f"Hasil: {label}")
    st.write(f"Tingkat keyakinan: {proba[idx]*100:.1f}%")
    st.write(f"Area cacat: {cacat*100:.1f}%  |  Grade: {grade}")