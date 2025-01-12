import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# 计算单像素对的权重
def compute_pixel_weight(diff, di, dj, sigma, alpha):
    diff_norm = np.sum(diff ** 2)
    exp_value = -diff_norm / (2 * sigma ** 2)
    exp_value = np.clip(exp_value, -100, 0)  # 限制指数范围，避免溢出
    color_similarity = np.exp(exp_value)
    spatial_similarity = np.exp(-(di ** 2 + dj ** 2) / (2 * alpha ** 2))
    return color_similarity * spatial_similarity

# 预计算权重矩阵（使用 csr_matrix 替代 lil_matrix）
def compute_weight_matrix_color_enhanced(image, sigma, alpha=0.5):
    height, width, _ = image.shape
    row_indices = []
    col_indices = []
    values = []
    flat_image = image.reshape(-1, 3)  # 展平为 (像素数, RGB)

    # 遍历像素和其邻域
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4邻域
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    n_idx = ni * width + nj
                    diff = flat_image[idx] - flat_image[n_idx]
                    weight = compute_pixel_weight(diff, di, dj, sigma, alpha)
                    if weight > 1e-5:  # 过滤掉极小权重
                        row_indices.append(idx)
                        col_indices.append(n_idx)
                        values.append(weight)

    weights = csr_matrix((values, (row_indices, col_indices)), shape=(height * width, height * width))
    return weights

# 随机游走降噪函数（优化）
def random_walk_denoising_color_enhanced(image, weights, iterations=20, tol=1e-3, beta=0.85, update_interval=10):
    height, width, channels = image.shape
    flat_image = image.reshape(-1, channels).astype(np.float32)
    
    for iter_num in range(iterations):
        # 使用稀疏矩阵快速计算更新
        weight_sum = weights.sum(axis=1).A1
        weight_sum[weight_sum == 0] = 1e-10  # 避免除以 0
        new_image = weights.dot(flat_image) / weight_sum[:, None]
        
        # 每隔 update_interval 次动态更新权重
        if iter_num % update_interval == 0:
            weights = compute_weight_matrix_color_enhanced(new_image.reshape(height, width, channels), sigma=8, alpha=1.2)
        
        # 增强通道间相关性
        flat_image = beta * new_image + (1 - beta) * flat_image  # 引入历史信息
        flat_image = np.clip(flat_image, 0, 255)  # 限制值范围
        if np.linalg.norm(new_image - flat_image) < tol:  # 收敛条件
            break
    return flat_image.reshape(height, width, channels).astype(np.uint8)

# 替换黑色像素
def replace_black_pixels_with_mean(image, threshold=15):
    """
    将图像中接近黑色的像素替换为其周围像素的均值。
    
    参数:
    - image: 输入图像 (numpy array)
    - threshold: 黑色判断的阈值，RGB 最大值低于该值即为黑色。
    
    返回:
    - 替换后的图像
    """
    height, width, channels = image.shape
    new_image = image.copy()
    
    # 遍历图像每个像素
    for i in range(height):
        for j in range(width):
            # 判断是否是接近黑色的点
            if np.max(image[i, j]) < threshold:
                # 获取 8 邻域坐标
                neighbors = []
                for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbors.append(image[ni, nj])
                
                # 计算邻域均值，忽略接近黑色的点
                neighbors = np.array([n for n in neighbors if np.max(n) >= threshold])
                if len(neighbors) > 0:
                    new_image[i, j] = np.mean(neighbors, axis=0)  # 用邻域均值替代
                else:
                    new_image[i, j] = [128, 128, 128]  # 如果无有效邻域，用灰色填充
    return new_image

# 加载图像路径
image_path = 'uniform_noise_colored.jpg'

# 加载彩色图像
image = cv2.imread(image_path)

# 降噪处理
weights = compute_weight_matrix_color_enhanced(image, sigma=8, alpha=1.2)
denoised_image = random_walk_denoising_color_enhanced(image, weights, iterations=25, beta=0.9, update_interval=10)

# 替换黑色像素
processed_image = replace_black_pixels_with_mean(denoised_image, threshold=15)

# 显示结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Denoised Image")
plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Processed Image (Black Replaced)")
plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()