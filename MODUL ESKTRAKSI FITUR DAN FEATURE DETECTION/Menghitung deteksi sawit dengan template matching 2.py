import cv2
import numpy as np

# Membaca gambar utuh untuk dicari
img_rgb = cv2.imread('keluarga semut.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Membaca template
template = cv2.imread('Semut.jpg', 0)

# Ukuran template. Ukuran ini akan digunakan untuk menggambar kotak
w, h = template.shape[::-1]

# Menggunakan metode COEFF-NORMALIZED
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

# Menentukan threshold atau ambang batas deteksi kemiripan titik.
# Lakukan eksperimen dengan merubah nilai ini
threshold = 0.5

# Mencari posisi dengan nilai di atas threshold
loc = np.where(res >= threshold)

# Menggunakan set untuk menyimpan koordinat yang unik
points = set()

for pt in zip(*loc[::-1]):
    # Menentukan area yang akan diabaikan dalam radius tertentu
    ignore_radius = 20
    is_duplicate = False

    for p in points:
        if abs(pt[0] - p[0]) < ignore_radius and abs(pt[1] - p[1]) < ignore_radius:
            is_duplicate = True
            break

    if not is_duplicate:
        # Gambar persegi warna kuning dengan ketebalan dua poin
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
        points.add(pt)

# Menampilkan jumlah objek yang ditemukan
count = len(points)
print("Jumlah objek ditemukan:", count)

# Tampilkan dengan imshow
cv2.imshow('Deteksi Objek', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
