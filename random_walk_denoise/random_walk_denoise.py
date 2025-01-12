import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from skimage.metrics import structural_similarity as ssim
import time

# 计算单像素对的权重
def compute_pixel_weight(diff, di, dj, sigma, alpha):
    diff_norm = np.sum(diff ** 2)
    exp_value = -diff_norm / (2 * sigma ** 2)
    exp_value = np.clip(exp_value, -100, 0)  
    color_similarity = np.exp(exp_value)
    spatial_similarity = np.exp(-(di ** 2 + dj ** 2) / (2 * alpha ** 2))
    return color_similarity * spatial_similarity

# 预计算权重矩阵（使用 csr_matrix 替代 lil_matrix）
def compute_weight_matrix_color_enhanced(image, sigma, alpha=0.5):
    height, width, _ = image.shape
    row_indices = []
    col_indices = []
    values = []
    flat_image = image.reshape(-1, 3)  

    for i in range(height):
        for j in range(width):
            idx = i * width + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    n_idx = ni * width + nj
                    diff = flat_image[idx] - flat_image[n_idx]
                    weight = compute_pixel_weight(diff, di, dj, sigma, alpha)
                    if weight > 1e-5: 
                        row_indices.append(idx)
                        col_indices.append(n_idx)
                        values.append(weight)

    weights = csr_matrix((values, (row_indices, col_indices)), shape=(height * width, height * width))
    return weights

# 随机游走降噪函数
def random_walk_denoising_color_enhanced(image, weights, iterations=10, tol=1e-3, beta=0.8, update_interval=5):
    height, width, channels = image.shape
    flat_image = image.reshape(-1, channels).astype(np.float32)
    
    for iter_num in range(iterations):
        # 使用稀疏矩阵快速计算更新
        weight_sum = weights.sum(axis=1).A1
        weight_sum[weight_sum == 0] = 1e-10  # 避免除以 0
        new_image = weights.dot(flat_image) / weight_sum[:, None]
        
        # 每隔 update_interval 次动态更新权重
        if iter_num % update_interval == 0:
            weights = compute_weight_matrix_color_enhanced(new_image.reshape(height, width, channels), sigma=10, alpha=1.5)
        
        # 增强通道间相关性
        flat_image = beta * new_image + (1 - beta) * flat_image  
        flat_image = np.clip(flat_image, 0, 255) 
        if np.linalg.norm(new_image - flat_image) < tol: 
            break
    return flat_image.reshape(height, width, channels).astype(np.uint8)

# 计算性能指标
def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def psnr(image1, image2):
    mse_value = mse(image1, image2)
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_value)) if mse_value > 0 else float('inf')

def calculate_ssim_multichannel(image1, image2):
    ssim_values = []
    for channel in range(image1.shape[2]):
        ssim_values.append(ssim(image1[:, :, channel], image2[:, :, channel], data_range=255))
    return np.mean(ssim_values)

def calculate_epi(original, filtered):
    diff = np.abs(original - filtered)
    return np.mean(diff)

image_path = 'gaussian_noise_colored.jpg'

image = cv2.imread(image_path)

params = [
    {"sigma": 5, "alpha": 1.0},   # 更低的平滑，适合轻噪声
    {"sigma": 8, "alpha": 1.2},   # 中等平滑，适合中噪声
    {"sigma": 10, "alpha": 1.5},  # 平衡平滑和细节
    {"sigma": 12, "alpha": 2.0}   # 更强平滑，适合较高噪声
]

plt.figure(figsize=(15, 10))


for i, param in enumerate(params):
    start_time = time.time()
    weights = compute_weight_matrix_color_enhanced(image, sigma=param["sigma"], alpha=param["alpha"])
    denoised_image = random_walk_denoising_color_enhanced(image, weights, iterations=15, beta=0.85, update_interval=5)
    execution_time = time.time() - start_time

    # 计算性能指标
    mse_value = mse(image, denoised_image)
    psnr_value = psnr(image, denoised_image)
    ssim_value = calculate_ssim_multichannel(image, denoised_image)
    epi_value = calculate_epi(image, denoised_image)

    print(f"Sigma={param['sigma']}, Alpha={param['alpha']}")
    print(f"MSE: {mse_value:.2f}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"EPI: {epi_value:.4f}")
    print(f"Execution Time: {execution_time:.4f} seconds")
    print("-" * 40)
    
    plt.subplot(2, 2, i + 1)
    plt.title(f"Sigma={param['sigma']}, Alpha={param['alpha']}")
    plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

plt.tight_layout()
plt.show()
