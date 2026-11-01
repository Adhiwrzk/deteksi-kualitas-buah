import os
import cv2
import matplotlib.pyplot as plt

FOLDER = "data/train"

# 1) Jumlah gambar per kelas
jumlah = {}
for kelas in sorted(os.listdir(FOLDER)):
    jumlah[kelas] = len(os.listdir(os.path.join(FOLDER, kelas)))
print("Jumlah gambar per kelas (train):")
for k, v in jumlah.items():
    print(f"  {k}: {v}")
print("Total:", sum(jumlah.values()))

# 2) Grafik jumlah per kelas
plt.figure(figsize=(9, 4))
plt.bar(jumlah.keys(), jumlah.values())
plt.xticks(rotation=45, ha="right")
plt.title("Jumlah gambar per kelas")
plt.tight_layout()
plt.savefig("grafik_jumlah_kelas.png", dpi=150)

# 3) Ukuran dan channel gambar (contoh dari tiap kelas)
print("\nUkuran gambar (tinggi x lebar x channel):")
for kelas in sorted(os.listdir(FOLDER)):
    nama = os.listdir(os.path.join(FOLDER, kelas))[0]
    img = cv2.imread(os.path.join(FOLDER, kelas, nama))
    print(f"  {kelas}: {img.shape}")

# 4) Contoh gambar tiap kelas (3 gambar per kelas)
kelas_list = sorted(os.listdir(FOLDER))
fig, axs = plt.subplots(len(kelas_list), 3, figsize=(8, 2.6 * len(kelas_list)))
for i, kelas in enumerate(kelas_list):
    files = os.listdir(os.path.join(FOLDER, kelas))[:3]
    for j, nama in enumerate(files):
        img = cv2.imread(os.path.join(FOLDER, kelas, nama))
        axs[i][j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i][j].axis("off")
        if j == 0:
            axs[i][j].set_title(kelas, loc="left", fontsize=10)
plt.tight_layout()
plt.savefig("contoh_gambar_kelas.png", dpi=150)
plt.show()