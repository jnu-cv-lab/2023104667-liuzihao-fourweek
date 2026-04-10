import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====================== 第二部分：σ公式验证（M=4） ======================

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
def gaussian_blur_then_downsample(img, M, sigma):
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return downsample(blurred, M)

# ====================== 实验参数设置 ======================
M = 4  # 固定下采样倍数
sigma_list = [0.5, 1.0, 2.0, 4.0]  # 实验σ值
theo_sigma = 0.45 * M  # 理论最优σ=1.8

# ====================== 生成测试图 ======================
checker = generate_checkboard(256, 8)
chirp = generate_chirp(256)

# ====================== 直接下采样（基准） ======================
checker_direct = downsample(checker, M)
chirp_direct = downsample(chirp, M)

# ====================== 可视化：Chirp图不同σ对比 ======================
plt.figure(figsize=(18, 8))
# 原始图 + 直接下采样
plt.subplot(2, 6, 1)
plt.imshow(chirp, cmap='gray')
plt.title('Original Chirp', fontsize=10)
plt.axis('off')

plt.subplot(2, 6, 2)
plt.imshow(chirp_direct, cmap='gray')
plt.title('Direct Downsample\n(Aliasing)', fontsize=10)
plt.axis('off')

# 不同σ滤波后下采样
for idx, sigma in enumerate(sigma_list):
    res = gaussian_blur_then_downsample(chirp, M, sigma)
    plt.subplot(2, 6, idx + 3)
    plt.imshow(res, cmap='gray')
    plt.title(f'σ={sigma}', fontsize=10)
    plt.axis('off')

# 理论最优σ
res_theo = gaussian_blur_then_downsample(chirp, M, theo_sigma)
plt.subplot(2, 6, 7)
plt.imshow(res_theo, cmap='gray')
plt.title(f'Theoretical Optimal\nσ={theo_sigma:.1f}', fontsize=10)
plt.axis('off')

plt.tight_layout()
plt.savefig('part2_chirp_sigma_result.png', dpi=300, bbox_inches='tight')
plt.show()

# ====================== 可视化：棋盘格图不同σ对比 ======================
plt.figure(figsize=(18, 8))
# 原始图 + 直接下采样
plt.subplot(2, 6, 1)
plt.imshow(checker, cmap='gray')
plt.title('Original Checkerboard', fontsize=10)
plt.axis('off')

plt.subplot(2, 6, 2)
plt.imshow(checker_direct, cmap='gray')
plt.title('Direct Downsample\n(Aliasing)', fontsize=10)
plt.axis('off')

# 不同σ滤波后下采样
for idx, sigma in enumerate(sigma_list):
    res = gaussian_blur_then_downsample(checker, M, sigma)
    plt.subplot(2, 6, idx + 3)
    plt.imshow(res, cmap='gray')
    plt.title(f'σ={sigma}', fontsize=10)
    plt.axis('off')

# 理论最优σ
res_theo_checker = gaussian_blur_then_downsample(checker, M, theo_sigma)
plt.subplot(2, 6, 7)
plt.imshow(res_theo_checker, cmap='gray')
plt.title(f'Theoretical Optimal\nσ={theo_sigma:.1f}', fontsize=10)
plt.axis('off')

plt.tight_layout()
plt.savefig('part2_checker_sigma_result.png', dpi=300, bbox_inches='tight')
plt.show()