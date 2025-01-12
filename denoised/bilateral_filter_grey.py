import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import time

def bilateral_filter_grey(image_array, sigma_s=2, sigma_r=0.1):
    """
    对灰度图像应用双边滤波。
    参数：
    - image_array: 输入图像的二维像素值数组
    - sigma_s: 空间域标准差（控制滤波器的空间范围）
    - sigma_r: 范围域标准差（控制像素值相似度的权重）
    返回：
    - 滤波后的图像数组
    """
    height, width = image_array.shape
    filtered_image = np.zeros_like(image_array)

    padded_image = np.pad(image_array, sigma_s, mode='reflect')
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i + 2 * sigma_s + 1, j:j + 2 * sigma_s + 1]
            spatial_weights = np.exp(-((np.arange(-sigma_s, sigma_s + 1)[:, None])**2 +
                                        (np.arange(-sigma_s, sigma_s + 1)[None, :])**2) / (2 * sigma_s**2))
            intensity_weights = np.exp(-((region - image_array[i, j])**2) / (2 * (sigma_r * 255)**2))
            weights = spatial_weights * intensity_weights
            filtered_image[i, j] = np.sum(weights * region) / np.sum(weights)
    
    return filtered_image.astype(np.uint8)

def mse(image1, image2):
    """计算均方误差 (MSE)"""
    return np.mean((image1 - image2) ** 2)

def psnr(image1, image2):
    """计算峰值信噪比 (PSNR)"""
    mse_value = mse(image1, image2)
    max_pixel = 255.0
    if mse_value == 0:  # 防止log(0)错误
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))

def calculate_epi(original, filtered):
    """
    计算增强像素差异 (EPI)
    """
    diff = np.abs(original - filtered)
    return np.mean(diff)

def calculate_metrics(original, filtered):
    """计算 MSE、PSNR、SSIM 和 EPI"""
    mse_value = mse(original, filtered)
    psnr_value = psnr(original, filtered)
    # 设置适配窗口大小
    win_size = min(7, original.shape[0], original.shape[1])  # 窗口大小不能超过图像尺寸
    ssim_value = ssim(original, filtered, win_size=win_size)
    epi_value = calculate_epi(original, filtered)
    return mse_value, psnr_value, ssim_value, epi_value

def main():
    # 创建保存路径
    output_dir = "denoised"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开输入图像
    image_path = "gamma_noise_grey.jpg"
    input_image = Image.open(image_path).convert('L')  # 转为灰度图像
    input_array = np.array(input_image)

    # 应用双边滤波
    sigma_s = 2  # 空间域标准差
    sigma_r = 0.1  # 范围域标准差
    start_time = time.time()
    filtered_array = bilateral_filter_grey(input_array, sigma_s, sigma_r)
    end_time = time.time()

    # 计算性能指标
    mse_value, psnr_value, ssim_value, epi_value = calculate_metrics(input_array, filtered_array)
    execution_time = end_time - start_time

    # 保存滤波后的图像
    output_path = os.path.join(output_dir, "gamma_noise_bilateral_filtered_grey.jpg")
    filtered_image = Image.fromarray(filtered_array)
    filtered_image.save(output_path)
    print(f"Filtered image saved to {output_path}")

    # 打印结果
    print(f"MSE: {mse_value:.2f}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"EPI: {epi_value:.4f}")
    print(f"Execution Time: {execution_time:.4f} seconds")

if __name__ == "__main__":
    main()
