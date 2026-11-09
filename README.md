# Deteksi Kualitas Buah Apel Berdasarkan Citra Digital

Project Akhir mata kuliah Pengolahan Citra Digital. Aplikasi mengklasifikasikan kondisi apel
menjadi tiga kelas (Bagus, Bercak, Busuk) dari sebuah foto, memakai Convolutional Neural Network (CNN).

## Dataset
Fruits Dataset for Fruit Disease Classification (Kaggle), bagian apel:
https://www.kaggle.com/datasets/ateebnoone/fruits-dataset-for-fruit-disease-classification

Struktur folder yang dibutuhkan notebook:
dataset/
  Bagus/
  Busuk/
  Bercak/

## File
- `deteksi_apel.ipynb` : notebook dengan tahapan CRISP-DM (pemahaman data, pra-pemrosesan, pemodelan, evaluasi).
- `app.py` : aplikasi web Streamlit.

## Cara menjalankan
1. Unduh dataset dari tautan di atas dan susun seperti struktur folder di atas.
2. Instal pustaka: `pip install tensorflow opencv-python matplotlib streamlit`
3. Jalankan `deteksi_apel.ipynb` sampai selesai. Notebook menghasilkan `model_kualitas_apel.h5`.
4. Jalankan aplikasi: `python -m streamlit run app.py`

Catatan: file model `.h5` tidak disertakan di repository karena ukurannya besar.# deteksi-kualitas-buah
