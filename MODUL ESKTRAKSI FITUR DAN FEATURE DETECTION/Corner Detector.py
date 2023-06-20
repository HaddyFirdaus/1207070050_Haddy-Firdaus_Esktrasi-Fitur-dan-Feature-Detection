import numpy as np
import cv2
from matplotlib import pyplot as plt

# Gunakan gambar
img = cv2.imread('Satoru.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Deteksi pojok dengan GFTT
corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 10)
corners = np.int0(corners)

# Menampilkan jumlah titik terdeteksi
print("Jumlah titik terdeteksi = ", corners.shape[0])

# Untuk ditampilkan di Matplotlib, urutan band dibalik
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perbesar ukuran hasil plotting
plt.rcParams["figure.figsize"] = (20, 20)

# Untuk tiap pojok yang terdeteksi, munculkan pada gambar
for i in corners:
    x, y = i.ravel()
    cv2.circle(rgb, (x, y), 3, (255, 0, 0), -1)

plt.imshow(rgb)
plt.show()
