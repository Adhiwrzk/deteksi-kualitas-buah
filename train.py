import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.pipeline import GRUP_DEFAULT, daftar_citra, muat_fitur, susun

if __name__ == "__main__":
    print("Mengekstrak fitur data latih...")
    p_tr, l_tr = daftar_citra("data/train")
    d_tr, y_tr = muat_fitur(p_tr, l_tr)
    print("Mengekstrak fitur data uji...")
    p_te, l_te = daftar_citra("data/test")
    d_te, y_te = muat_fitur(p_te, l_te)

    X_tr, X_te = susun(d_tr, GRUP_DEFAULT), susun(d_te, GRUP_DEFAULT)
    print(f"Fitur per citra: {X_tr.shape[1]} | latih: {len(y_tr)} | uji: {len(y_te)}")

    model = make_pipeline(StandardScaler(),
                          SVC(kernel="rbf", C=10, probability=True, class_weight="balanced"))
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    laporan = classification_report(y_te, pred, digits=4)
    print("Akurasi:", round(accuracy_score(y_te, pred), 4))
    print(laporan)
    print(confusion_matrix(y_te, pred, labels=model.classes_))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_svm.pkl")
    with open("laporan_klasifikasi.txt", "w", encoding="utf-8") as f:
        f.write(laporan)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, labels=model.classes_,
                                            xticks_rotation=45, ax=ax, colorbar=False)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Tersimpan: models/model_svm.pkl, laporan_klasifikasi.txt, confusion_matrix.png")
