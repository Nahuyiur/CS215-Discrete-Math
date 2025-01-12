import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import time

# 计算双边滤波的权重矩阵
def compute_bilateral_weights(image, sigma_s, sigma_r):
    h, w = image.shape
    num_pixels = h * w
    flat_image = image.flatten()
    row_indices = []
    col_indices = []
    values = []

    for i in range(h):
        for j in range(w):
            p_idx = i * w + j
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        q_idx = ni * w + nj
                        spatial_diff = di**2 + dj**2
                        intensity_diff = (flat_image[p_idx] - flat_image[q_idx])**2
                        weight = np.exp(-spatial_diff / (2 * sigma_s**2)) * \
                                 np.exp(-intensity_diff / (2 * sigma_r**2))
                        row_indices.append(p_idx)
                        col_indices.append(q_idx)
                        values.append(weight)

    W = sp.csr_matrix((values, (row_indices, col_indices)), shape=(num_pixels, num_pixels))
    D = sp.diags(W.sum(axis=1).A1).tocsc()
    return W, D

# 计算性能指标
def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def psnr(image1, image2):
    mse_value = mse(image1, image2)
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_value)) if mse_value > 0 else float('inf')

def calculate_ssim(image1, image2):
    return ssim(image1, image2, data_range=255, multichannel=True)

def calculate_epi(original, filtered):
    diff = np.abs(original - filtered)
    return np.mean(diff)

# 预条件共轭梯度法 (PCG)
def pcg_denoising(W, D, b, original_image, tol=1e-6, max_iter=50):
    L = D - W  
    x = np.zeros_like(b, dtype=np.float64) 
    r = b - L @ x
    z = spla.spsolve(D, r)  
    p = z
    rs_old = r @ z

    for i in range(max_iter):
        Ap = L @ p
        alpha = rs_old / (p @ Ap)
        x += alpha * p 
        r -= alpha * Ap
        z = spla.spsolve(D, r)
        rs_new = r @ z

        if np.sqrt(rs_new) < tol:
            print(f"PCG 收敛于第 {i+1} 次迭代")
            break
        p = z + (rs_new / rs_old) * p
        rs_old = rs_new

    return x.clip(0, 255) 

# Nesterov 加速法
def nesterov_denoising(W, D, b, original_image, max_iter=50):
    x = np.zeros_like(b, dtype=np.float64) 
    y = np.zeros_like(b, dtype=np.float64)
    t = 1

    for k in range(1, max_iter + 1):
        # 预测步
        x_old = x
        x = y - (D - W) @ y
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        beta = (t - 1) / t_new

        # 更新步
        y = x + beta * (x - x_old)
        t = t_new

    return x.clip(0, 255)

# 主程序
if __name__ == "__main__":
    image_path = "output_images/1.jpg" 
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("无法加载输入图像，请检查路径！")

    h, w, c = image.shape
    denoised_channels_pcg = []
    denoised_channels_nesterov = []

    for channel in range(c):
        print(f"\n处理通道 {channel + 1}/{c}...")
        channel_image = image[:, :, channel]
        b = channel_image.flatten()

        # 计算权重矩阵和度矩阵
        print("计算权重矩阵和度矩阵...")
        start_time = time.time()
        W, D = compute_bilateral_weights(channel_image, sigma_s=5, sigma_r=20)
        print(f"权重矩阵计算完成，耗时 {time.time() - start_time:.2f} 秒")

        # 使用 PCG 方法
        print("\n使用 PCG 方法进行降噪...")
        start_time = time.time()
        pcg_result = pcg_denoising(W, D, b, channel_image)
        print(f"PCG 方法完成，耗时 {time.time() - start_time:.2f} 秒")
        denoised_channels_pcg.append(pcg_result.reshape(h, w))

        # 使用 Nesterov 方法
        print("\n使用 Nesterov 方法进行降噪...")
        start_time = time.time()
        nesterov_result = nesterov_denoising(W, D, b, channel_image)
        print(f"Nesterov 方法完成，耗时 {time.time() - start_time:.2f} 秒")
        denoised_channels_nesterov.append(nesterov_result.reshape(h, w))

    # 合并各通道结果
    pcg_image = np.stack(denoised_channels_pcg, axis=2).astype(np.uint8)
    nesterov_image = np.stack(denoised_channels_nesterov, axis=2).astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.title("Original Noisy Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("PCG Denoised Image")
    plt.imshow(cv2.cvtColor(pcg_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Nesterov Denoised Image")
    plt.imshow(cv2.cvtColor(nesterov_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()
