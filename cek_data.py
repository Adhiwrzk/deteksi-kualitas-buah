import os, cv2, hashlib

jumlah = rusak = buram = kecil = duplikat = 0
hash_terlihat = set()
for folder in ["data/train", "data/test"]:
    for kelas in os.listdir(folder):
        for nama in os.listdir(os.path.join(folder, kelas)):
            p = os.path.join(folder, kelas, nama)
            jumlah += 1
            img = cv2.imread(p)
            if img is None:
                rusak += 1
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
                buram += 1
            if min(img.shape[:2]) < 100:
                kecil += 1
            h = hashlib.md5(open(p, "rb").read()).hexdigest()
            if h in hash_terlihat:
                duplikat += 1
            hash_terlihat.add(h)

print("Total berkas:", jumlah)
print("Tidak dapat dibaca:", rusak)
print("Kemungkinan buram:", buram, "(ambang 50, bersifat perkiraan)")
print("Resolusi sangat kecil (<100 px):", kecil)
print("Berkas identik (duplikat persis):", duplikat)