# tampilkan kedua gambar 
import cv2
from matplotlib import pyplot as plt 
# panggil dan konversi warna agar sesuai dengan Matplotlib 
semut = cv2.imread('semut.jpg') 
semut = cv2.cvtColor(semut, cv2.COLOR_BGR2RGB) 
# panggil dan konversi warna agar sesuai dengan Matplotlib 
keluarga_semut = cv2.imread('keluarga semut.jpg') 
keluarga_semut = cv2.cvtColor(keluarga_semut, cv2.COLOR_BGR2RGB) 
plt.subplot(121),plt.imshow(semut), plt.title('semut') 
plt.subplot(122),plt.imshow(keluarga_semut), plt.title('keluarga semut') 
plt.show()