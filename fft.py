import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def DFT2D(image):
    M, N = image.shape
    dft2d = np.zeros((M, N), dtype=complex)
    u = np.arange(M).reshape(M, 1)
    v = np.arange(N).reshape(1, N)
    for x in range(M):
        for y in range(N):
            dft2d += image[x, y] * np.exp(-2j * np.pi * ((u * x / M) + (v * y / N)))
    return dft2d

# 画像をグレースケールで読み込む
# img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
img = Image.open('lena.png')
# グレイスケールに変換する
gray_img = img.convert('L')
# NumPy 配列にする
f_xy = np.asarray(gray_img)
# 2次元離散フーリエ変換 (2D-DFT) を行う
# dft = DFT2D(img)
dft = DFT2D(f_xy)

# 低周波成分が画像の中心にくるようにシフト
dft_shift = np.fft.fftshift(dft)

# スペクトルの大きさを計算
magnitude_spectrum = 20 * np.log1p(np.abs(dft_shift))

# 元の画像とスペクトルを表示
# plt.subplot(121), plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()
# NumPy配列からPIL Imageを作成
magnitude_spectrum_normalized = (magnitude_spectrum - np.min(magnitude_spectrum)) / np.ptp(magnitude_spectrum) * 255
magnitude_spectrum_image = Image.fromarray(magnitude_spectrum_normalized.astype(np.uint8))

# 画像をPNG形式で保存
magnitude_spectrum_image.save('spectrum.png')
# magnitude_spectrum.save('./output.png', 'PNG')