import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====================== 第一部分：下采样与混叠验证 ======================

# 1. 生成棋盘格图像
def generate_checkboard(size=256, block_size=8):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i // block_size + j // block_size) % 2 == 0:
                img[i, j] = 255
    return img

# 2. 生成 chirp 频率渐变图像
def generate_chirp(size=256):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    chirp = np.sin(np.pi * R * size / 2)
    chirp = ((chirp - chirp.min()) / (chirp.max() - chirp.min()) * 255).astype(np.uint8)
    return chirp

# 3. 直接下采样 M 倍
def downsample(img, M):
    return img[::M, ::M]

# 4. 高斯滤波后下采样
def gaussian_blur_then_downsample(img, M=4, sigma=1.8):
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return downsample(blurred, M)

# 5. 计算并中心化 FFT 频谱
def fft_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return 20 * np.log(np.abs(fshift) + 1)

# ====================== 生成图像 ======================
checker = generate_checkboard(256, 8)
chirp = generate_chirp(256)

# ====================== 直接下采样 ======================
M = 4
checker_direct = downsample(checker, M)
chirp_direct = downsample(chirp, M)

# ====================== 滤波后下采样 ======================
checker_blur_ds = gaussian_blur_then_downsample(checker, M, sigma=1.8)
chirp_blur_ds = gaussian_blur_then_downsample(chirp, M, sigma=1.8)

# ====================== 计算频谱 ======================
spec_chirp_ori = fft_spectrum(chirp)
spec_chirp_direct = fft_spectrum(chirp_direct)
spec_chirp_blur = fft_spectrum(chirp_blur_ds)

# ====================== 显示结果 ======================
plt.figure(figsize=(16, 12))

# 棋盘格
plt.subplot(3, 4, 1)
plt.imshow(checker, cmap='gray')
plt.title('Original Checkerboard')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(checker_direct, cmap='gray')
plt.title('Direct Downsample (Aliasing)')
plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(checker_blur_ds, cmap='gray')
plt.title('Gaussian + Downsample')
plt.axis('off')

# Chirp
plt.subplot(3, 4, 5)
plt.imshow(chirp, cmap='gray')
plt.title('Original Chirp')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(chirp_direct, cmap='gray')
plt.title('Direct Downsample (Aliasing)')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(chirp_blur_ds, cmap='gray')
plt.title('Gaussian + Downsample')
plt.axis('off')

# 频谱
plt.subplot(3, 4, 9)
plt.imshow(spec_chirp_ori, cmap='gray')
plt.title('Chirp Spectrum (Original)')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(spec_chirp_direct, cmap='gray')
plt.title('Spectrum (Direct Downsample)')
plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(spec_chirp_blur, cmap='gray')
plt.title('Spectrum (Gaussian + Downsample)')
plt.axis('off')

plt.tight_layout()
plt.show()