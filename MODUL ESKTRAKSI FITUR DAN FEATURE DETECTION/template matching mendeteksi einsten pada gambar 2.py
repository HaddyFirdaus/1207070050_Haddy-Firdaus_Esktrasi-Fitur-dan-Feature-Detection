import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('akatsuki.jpg', 0)
img_rgb2 = img_rgb.copy()
template = cv2.imread('itachi.jpg', 0)
h, w = template.shape

# All the 6 methods for comparison in a list
methods = [
    cv2.TM_CCOEFF,
    cv2.TM_CCOEFF_NORMED,
    cv2.TM_CCORR,
    cv2.TM_CCORR_NORMED,
]

# Perbesar ukuran hasil plotting
plt.rcParams["figure.figsize"] = (15, 15)

for method in methods:
    img = img_rgb2.copy()

    # Menggunakan template matching
    res = cv2.matchTemplate(img, template, method)

    # Mencari ukuran citra template untuk menggambar kotak
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Metode TM_SQDIFF dan TM_SQDIFF_NORMED menggunakan persamaan yang sedikit berbeda
    # Sehingga dibuatkan fungsi khusus untuk mengambil nilai minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Buat persegi pada lokasi yang ditemukan
    cv2.rectangle(img, top_left, bottom_right, 255, 2)  # 2 adalah ketebalan garis kotak

    print("Hasil metode", method, ":")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(res, cmap='gray')
    ax1.set_title('Hasil matching')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(img, cmap='gray')
    ax2.set_title('Lokasi terdeteksi')
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.show()
