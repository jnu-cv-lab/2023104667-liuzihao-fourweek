import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 1. Configuration ----------------------
img_path = "/home/hhhkinggoder1/cv-course/homework4/beauty.jpg"
block_size = 16
energy_ratio = 0.95

# Read grayscale image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape
h = (h // block_size) * block_size
w = (w // block_size) * block_size
img = img[:h, :w]

# ---------------------- 2. Gradient Frequency ----------------------
def grad_freq(block):
    block = block.astype(np.float32)
    gx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = gx**2 + gy**2
    E_grad2 = np.mean(mag2)
    var_I = np.var(block)
    if var_I < 1e-8:
        return 0
    f_rms = np.sqrt(E_grad2 / (4 * np.pi**2 * var_I))
    return np.clip(f_rms / (block_size/2), 0, 0.5)

# ---------------------- 3. FFT 95% Energy Frequency ----------------------
def fft_95_freq(block):
    block = block.astype(np.float32) - np.mean(block)
    f = np.fft.fft2(block)
    f_shift = np.fft.fftshift(f)
    ps = np.abs(f_shift)**2
    total = np.sum(ps)
    if total < 1e-8:
        return 0
    cy, cx = block_size//2, block_size//2
    y, x = np.ogrid[:block_size, :block_size]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    idx_sorted = np.argsort(dist.flatten())
    cum_energy = 0
    for idx in idx_sorted:
        cum_energy += ps.flatten()[idx]
        if cum_energy >= energy_ratio * total:
            return dist.flatten()[idx] / block_size
    return 0.5

# ---------------------- 4. Compute all frequencies ----------------------
grad_list = []
fft_list = []
for i in range(0, h, block_size):
    for j in range(0, w, block_size):
        block = img[i:i+block_size, j:j+block_size]
        grad_list.append(grad_freq(block))
        fft_list.append(fft_95_freq(block))

# ---------------------- 5. Plot Histograms ----------------------
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.hist(fft_list, bins=20, color='red', alpha=0.7)
plt.title("FFT 95% Max Frequency Distribution")
plt.xlabel("Normalized Frequency (0~0.5)")
plt.ylabel("Block Count")
plt.xlim(0, 0.5)

plt.subplot(122)
plt.hist(grad_list, bins=20, color='blue', alpha=0.7)
plt.title("Spatial Gradient Frequency Distribution")
plt.xlabel("Normalized Frequency (0~0.5)")
plt.ylabel("Block Count")
plt.xlim(0, 0.5)

plt.tight_layout()
plt.show()

# Correlation
corr = np.corrcoef(grad_list, fft_list)[0, 1]
print(f"Correlation: {corr:.4f}")