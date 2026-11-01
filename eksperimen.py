"""Eksperimen perbandingan: (1) kombinasi grup fitur, (2) filter penghalusan.

Contoh:
    python eksperimen.py                 -> ablasi fitur, 400 citra per kelas
    python eksperimen.py --maks 0        -> memakai seluruh data (lebih lama)
    python eksperimen.py --blur          -> tambahkan perbandingan filter penghalusan
"""
import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.pipeline import daftar_citra, muat_fitur, susun

SKENARIO = {
    "A. Warna saja": ("warna",),
    "B. Warna + tekstur + cacat": ("warna", "tekstur", "cacat"),
    "C. B + statistik histogram": ("warna", "tekstur", "cacat", "statistik"),
    "D. B + tepi": ("warna", "tekstur", "cacat", "tepi"),
    "E. Semua fitur": ("warna", "tekstur", "cacat", "statistik", "tepi"),
}


def evaluasi(X_tr, y_tr, X_te, y_te):
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, class_weight="balanced"))
    t0 = time.time()
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return (accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro"),
            X_tr.shape[1], time.time() - t0)


def cetak(judul, baris):
    print(f"\n{judul}")
    print(f"{'Skenario':32s} {'Fitur':>6s} {'Akurasi':>9s} {'F1-macro':>9s} {'Waktu(s)':>9s}")
    for nama, (acc, f1, n, t) in baris.items():
        print(f"{nama:32s} {n:6d} {acc:9.4f} {f1:9.4f} {t:9.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--maks", type=int, default=400, help="citra per kelas (0 = semua)")
    ap.add_argument("--blur", action="store_true", help="bandingkan filter penghalusan")
    a = ap.parse_args()
    maks = a.maks or None

    p_tr, l_tr = daftar_citra(f"{a.data}/train", maks)
    p_te, l_te = daftar_citra(f"{a.data}/test", maks)

    print("Ekstraksi fitur (Gaussian blur)...")
    d_tr, y_tr = muat_fitur(p_tr, l_tr, "gaussian")
    d_te, y_te = muat_fitur(p_te, l_te, "gaussian")
    print(f"Latih: {len(y_tr)} citra | Uji: {len(y_te)} citra")

    hasil1 = {}
    for nama, grup in SKENARIO.items():
        hasil1[nama] = evaluasi(susun(d_tr, grup), y_tr, susun(d_te, grup), y_te)
    cetak("HASIL 1 - Perbandingan kombinasi fitur", hasil1)

    with open("hasil_eksperimen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eksperimen", "skenario", "jumlah_fitur", "akurasi", "f1_macro", "waktu_detik"])
        for nama, (acc, f1, n, t) in hasil1.items():
            w.writerow(["fitur", nama, n, round(acc, 4), round(f1, 4), round(t, 1)])

    plt.figure(figsize=(8, 4))
    plt.bar([n.split(".")[0] for n in hasil1], [v[0] for v in hasil1.values()])
    plt.ylim(min(v[0] for v in hasil1.values()) - 0.05, 1.0)
    plt.ylabel("Akurasi")
    plt.title("Perbandingan kombinasi fitur")
    plt.tight_layout()
    plt.savefig("grafik_perbandingan_fitur.png", dpi=150)

    if a.blur:
        hasil2 = {}
        grup = SKENARIO["E. Semua fitur"]
        for jenis in ["none", "box", "gaussian", "median"]:
            if jenis == "gaussian":
                dtr, ytr, dte, yte = d_tr, y_tr, d_te, y_te
            else:
                print(f"Ekstraksi fitur (blur = {jenis})...")
                dtr, ytr = muat_fitur(p_tr, l_tr, jenis)
                dte, yte = muat_fitur(p_te, l_te, jenis)
            hasil2[f"Filter: {jenis}"] = evaluasi(susun(dtr, grup), ytr, susun(dte, grup), yte)
        cetak("HASIL 2 - Perbandingan filter penghalusan (semua fitur)", hasil2)
        with open("hasil_eksperimen.csv", "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for nama, (acc, f1, n, t) in hasil2.items():
                w.writerow(["blur", nama, n, round(acc, 4), round(f1, 4), round(t, 1)])

    print("\nTersimpan: hasil_eksperimen.csv, grafik_perbandingan_fitur.png")
    print("Catatan: sebagian citra uji kemungkinan versi rotasi dari citra latih, "
          "jadi akurasi bisa tampak lebih tinggi dari kondisi nyata.")
