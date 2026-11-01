"""Pipeline pengolahan citra untuk deteksi kualitas buah (versi 3).

Tahapan: resize (INTER_AREA) -> filter penghalusan -> segmentasi buah ->
ekstraksi fitur (warna, tekstur, cacat, statistik histogram, tepi).
"""
import os
import random

import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import graycomatrix, graycoprops

UKURAN = (200, 200)

# Urutan grup fitur pada vektor akhir
NAMA_GRUP = ["warna", "tekstur", "cacat", "statistik", "tepi"]
GRUP_DEFAULT = tuple(NAMA_GRUP)


# ---------------------------------------------------------------- pra-pemrosesan
def preprocess(img_bgr, blur="gaussian"):
    """Resize 200x200 (INTER_AREA untuk memperkecil) lalu filter penghalusan.

    blur: "gaussian" | "box" | "median" | "none"
    """
    img = cv2.resize(img_bgr, UKURAN, interpolation=cv2.INTER_AREA)
    if blur == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), 0)
    if blur == "box":
        return cv2.blur(img, (5, 5))
    if blur == "median":
        return cv2.medianBlur(img, 5)
    return img


# ---------------------------------------------------------------- segmentasi
def segmentasi(img, T=25):
    """Latar = piksel mendekati putih (atau hitam di sudut rotasi) yang tersambung ke tepi."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    jarak_putih = np.linalg.norm(lab - np.array([255, 128, 128], np.float32), axis=2)
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    kandidat = ((jarak_putih < T) | (v < 40)).astype(np.uint8)
    _, lbl = cv2.connectedComponents(kandidat, connectivity=4)
    tepi = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
    tepi.discard(0)
    latar = np.isin(lbl, list(tepi))
    fg = (~latar).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    kontur, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(fg)
    if kontur:
        cv2.drawContours(mask, [max(kontur, key=cv2.contourArea)], -1, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask), mask


def mask_dalam(mask):
    """Bagian dalam buah (tepi dikikis) agar bayangan dan garis tepi objek tidak terhitung."""
    return cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=2)


# ---------------------------------------------------------------- fitur
def area_cacat(img, mask):
    """Cacat = piksel gelap atau keputihan (jamur) di bagian dalam buah."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    dalam = mask_dalam(mask)
    cacat = (((v < 80) | ((s < 60) & (v > 150))) & (dalam > 0)).astype(np.uint8) * 255
    cacat = cv2.morphologyEx(cacat, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    rasio = (cacat > 0).sum() / max((dalam > 0).sum(), 1)
    return cacat, rasio


def fitur_warna(hasil, mask):
    """Histogram warna HSV 8x4x4 bin, dinormalisasi h_i = n_i / n (jumlah = 1)."""
    hsv = cv2.cvtColor(hasil, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return hist / max(hist.sum(), 1)


def fitur_tekstur(hasil):
    """GLCM: contrast, homogeneity, energy, correlation (jarak 1, sudut 0 dan 90 derajat)."""
    gray = cv2.cvtColor(hasil, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0, np.pi / 2], levels=256, symmetric=True, normed=True)
    return np.array([graycoprops(glcm, p).mean()
                     for p in ("contrast", "homogeneity", "energy", "correlation")])


def fitur_statistik(gray, mask):
    """Mean, variansi, dan standar deviasi dari histogram ternormalisasi (area buah)."""
    hist = cv2.calcHist([gray], [0], mask, [256], [0, 256]).flatten()
    h = hist / max(hist.sum(), 1)
    k = np.arange(256)
    mean = (k * h).sum()
    var = (((k - mean) ** 2) * h).sum()
    return np.array([mean, var, np.sqrt(var)])


def fitur_tepi(gray, mask):
    """Kepadatan tepi Canny dan rata-rata gradien Sobel di bagian dalam buah."""
    dalam = mask_dalam(mask) > 0
    luas = max(dalam.sum(), 1)
    canny = cv2.Canny(gray, 50, 150)
    kepadatan = ((canny > 0) & dalam).sum() / luas
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    sobel = mag[dalam].mean() if dalam.any() else 0.0
    return np.array([kepadatan, sobel])


def ekstrak_semua(img_bgr, blur="gaussian"):
    """Hitung semua grup fitur sekali jalan. Mengembalikan (dict_fitur, hasil, mask, rasio_cacat)."""
    img = preprocess(img_bgr, blur)
    hasil, mask = segmentasi(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, rasio = area_cacat(img, mask)
    fitur = {
        "warna": fitur_warna(hasil, mask),
        "tekstur": fitur_tekstur(hasil),
        "cacat": np.array([rasio]),
        "statistik": fitur_statistik(gray, mask),
        "tepi": fitur_tepi(gray, mask),
    }
    return fitur, hasil, mask, rasio


def gabung(fitur, grup=GRUP_DEFAULT):
    return np.concatenate([fitur[g] for g in NAMA_GRUP if g in grup])


def ekstrak_fitur(img_bgr, grup=GRUP_DEFAULT, blur="gaussian"):
    """Dipakai aplikasi: (vektor_fitur, hasil_segmentasi, mask, rasio_cacat)."""
    fitur, hasil, mask, rasio = ekstrak_semua(img_bgr, blur)
    return gabung(fitur, grup), hasil, mask, rasio


# ---------------------------------------------------------------- dataset
def daftar_citra(folder, maks_per_kelas=None, seed=42):
    """Daftar path citra dan label (nama folder kelas)."""
    paths, labels = [], []
    rng = random.Random(seed)
    for kelas in sorted(os.listdir(folder)):
        dir_kelas = os.path.join(folder, kelas)
        if not os.path.isdir(dir_kelas):
            continue
        nama = sorted(os.listdir(dir_kelas))
        if maks_per_kelas:
            rng.shuffle(nama)
            nama = nama[:maks_per_kelas]
        for n in nama:
            paths.append(os.path.join(dir_kelas, n))
            labels.append(kelas)
    return paths, labels


def _fitur_satu(path, blur):
    img = cv2.imread(path)
    if img is None:
        return None
    return ekstrak_semua(img, blur)[0]


def muat_fitur(paths, labels, blur="gaussian", n_jobs=-1):
    """Ekstraksi fitur paralel. Berkas yang tidak terbaca dilewati."""
    hasil = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_fitur_satu)(p, blur) for p in paths)
    dicts, y = [], []
    for d, lab in zip(hasil, labels):
        if d is not None:
            dicts.append(d)
            y.append(lab)
    return dicts, np.array(y)


def susun(dicts, grup=GRUP_DEFAULT):
    """Susun daftar dict fitur menjadi matriks X."""
    return np.array([gabung(d, grup) for d in dicts])
