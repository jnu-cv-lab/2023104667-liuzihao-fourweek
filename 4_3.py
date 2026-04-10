import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ===================== 工具函数 =====================
def generate_checkboard(size=256, block_size=8):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i // block_size + j // block_size) % 2 == 0:
                img[i, j] = 255
    return img

def generate_chirp(size=256):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    chirp = np.sin(np.pi * R * size / 2)
    chirp = ((chirp - chirp.min()) / (chirp.max() - chirp.min()) * 255).astype(np.uint8)
    return chirp

def downsample(img, M=4):
    return img[::M, ::M]

def gaussian_blur(img, sigma):
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

# 计算梯度（判断细节多少）
def compute_gradient(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return grad

# ===================== 主流程 =====================
M = 4
img = generate_checkboard(256, 16)  # 你也可以换成 chirp 或自己的图
# img = generate_chirp(256)

# 1. 计算梯度
grad = compute_gradient(img)

# 2. 分区域：高梯度 / 低梯度
high_grad = grad > 60    # 细节多 → 小 σ
low_grad = grad <= 60    # 细节少 → 大 σ

# 3. 自适应 σ
sigma_small = 1.0   # 细节区
sigma_large = 3.0   # 背景区

# 4. 分别滤波
blur_small = gaussian_blur(img, sigma_small)
blur_large = gaussian_blur(img, sigma_large)

# 5. 融合自适应滤波图
blur_adaptive = np.zeros_like(img, dtype=np.uint8)
blur_adaptive[high_grad] = blur_small[high_grad]
blur_adaptive[low_grad] = blur_large[low_grad]

# 6. 统一滤波（对比用）
blur_uniform = gaussian_blur(img, 1.8)

# 7. 下采样
down_adaptive = downsample(blur_adaptive, M)
down_uniform = downsample(blur_uniform, M)

# 8. 误差图
error = cv2.absdiff(down_adaptive, down_uniform)

# ===================== 显示 =====================
plt.figure(figsize=(16, 8))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(grad, cmap='gray')
plt.title('Gradient Map')
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(blur_adaptive, cmap='gray')
plt.title('Adaptive Blur')
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(down_uniform, cmap='gray')
plt.title('Uniform σ=1.8')
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(down_adaptive, cmap='gray')
plt.title('Adaptive Downsample')
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(error, cmap='gray')
plt.title('Error Map')
plt.axis('off')

plt.tight_layout()
plt.savefig("part3_adaptive_result.png", dpi=300, bbox_inches='tight')
plt.show()