import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ===================== 工具函数（修复版） =====================
def compute_gradient(img):
    """计算梯度（判断细节多少）"""
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad = cv2.normalize(grad, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return grad

def downscale_region(img, M):
    """按 M 倍下采样，修复尺寸为0的问题"""
    h, w = img.shape
    # 强制保证新尺寸至少为1，避免报错
    new_h = max(1, int(h / M))
    new_w = max(1, int(w / M))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def upscale_back(img, target_h, target_w):
    """放大回原来大小，方便拼接"""
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

# ===================== 主流程：进阶版 —— 梯度估计局部 M（修复版） =====================
img = cv2.imread("face.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("找不到 face.jpg，请把人脸图放在代码文件夹！")
    exit()

H, W = img.shape

# 1. 计算梯度图
grad = compute_gradient(img)

# 2. 根据梯度 → 计算每个位置的 局部 M 值（核心！修复M范围）
# 规则：梯度大（细节多）→ M 小（少下采样，保清晰）；梯度小（背景）→ M 大（多下采样，抗混叠）
M_min = 2   # 细节区最多下采样 2 倍（最小M，保证下采样后尺寸≥1）
M_max = 4   # 背景区最多下采样 4 倍（降低最大M，避免边缘块尺寸为0）
local_M = M_max - (M_max - M_min) * grad  # 梯度越大，M 越小
# 强制限制M范围，避免异常值
local_M = np.clip(local_M, M_min, M_max)

# 3. 分块处理：每个区域用自己的 M 下采样（优化块大小，避免边缘块过小）
output = np.zeros_like(img, dtype=np.float32)
block_size = 32  # 增大块大小，避免边缘块尺寸异常

for y in range(0, H, block_size):
    for x in range(0, W, block_size):
        # 取出小块，处理边缘块
        y2 = min(y + block_size, H)
        x2 = min(x + block_size, W)
        block = img[y:y2, x:x2]
        block_h, block_w = block.shape
        
        # 跳过尺寸过小的块（避免M计算异常）
        if block_h < 4 or block_w < 4:
            output[y:y2, x:x2] = block
            continue
        
        # 这块的平均 M，强制取整并限制范围
        M_block = np.mean(local_M[y:y2, x:x2])
        M_block = int(round(M_block))
        M_block = max(M_min, min(M_block, M_max))  # 二次限制
        
        # 自适应 σ：0.45*M（理论公式）
        sigma = 0.45 * M_block
        ksize = int(6 * sigma + 1)
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        block_blur = cv2.GaussianBlur(block, (ksize, ksize), sigma)
        
        # 下采样 + 放大回来
        block_down = downscale_region(block_blur, M_block)
        block_restore = upscale_back(block_down, block_h, block_w)
        
        # 放回原图
        output[y:y2, x:x2] = block_restore

output = np.clip(output, 0, 255).astype(np.uint8)

# 对比：统一 M=4 下采样（和前两部分保持一致）
uniform_sigma = 1.8
uniform_blur = cv2.GaussianBlur(img, (int(6*uniform_sigma+1),)*2, uniform_sigma)
uniform_down = downscale_region(uniform_blur, 4)
uniform_final = upscale_back(uniform_down, H, W)

# 误差图
error = cv2.absdiff(output, uniform_final)

# ===================== 显示 =====================
plt.figure(figsize=(18, 10))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original Face')
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(grad, cmap='gray')
plt.title('Gradient Map')
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(local_M, cmap='jet')
plt.title(f'Local M (Bright=Small M={M_min}, Dark=Big M={M_max})')
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(uniform_final, cmap='gray')
plt.title('Uniform M=4, σ=1.8')
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(output, cmap='gray')
plt.title('Adaptive Local M Downsampling')
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(error, cmap='gray')
plt.title('Error Map (Adaptive vs Uniform)')
plt.axis('off')

plt.tight_layout()
plt.savefig("part3_advanced_local_M_fixed.png", dpi=300, bbox_inches='tight')
plt.show()