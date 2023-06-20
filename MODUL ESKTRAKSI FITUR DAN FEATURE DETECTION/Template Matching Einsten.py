#tampilkan kedua gambar 
import cv2
from matplotlib import pyplot as plt 
# panggil dan konversi warna agar sesuai dengan Matplotlib 
itachi = cv2.imread('itachi.jpg') 
itachi = cv2.cvtColor(itachi, cv2.COLOR_BGR2RGB) # simpan dengan nama yang sama = ditumpuk 
# # panggil dan konversi warna agar sesuai dengan Matplotlib 
akatsuki = cv2.imread('akatsuki.jpg') 
akatsuki = cv2.cvtColor(akatsuki, cv2.COLOR_BGR2RGB) 
plt.subplot(121),plt.imshow(itachi), plt.title('Itachi') 
plt.subplot(122),plt.imshow(akatsuki), plt.title('Akatsuki') 
plt.show()

