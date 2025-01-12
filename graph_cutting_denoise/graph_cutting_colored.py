import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim


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


def calculate_ssim_multichannel(image1, image2):
    """计算多通道图像的 SSIM"""
    ssim_values = []
    for channel in range(image1.shape[2]):
        ssim_value = ssim(image1[:, :, channel], image2[:, :, channel], data_range=image2.max() - image2.min())
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)


def calculate_epi(original, filtered):
    """计算增强像素差异 (EPI)"""
    diff = np.abs(original - filtered)
    return np.mean(diff)


def image_denoising_bk_color_optimized(image, data_weight=1000, smooth_weight=10, sigma_color=30, sigma_space=5):
    """
    改进的彩色图像降噪 - 基于最大流-最小割算法 (优化参数)
    """
    h, w, c = image.shape
    num_pixels = h * w
    source = num_pixels
    sink = num_pixels + 1

    # 创建有向图
    graph = nx.DiGraph()

    # 像素索引转换
    def pixel_index(x, y):
        return x * w + y

    # 转换图像为 float64
    image = image.astype(np.float64)

    # 添加数据项
    for i in range(h):
        for j in range(w):
            idx = pixel_index(i, j)
            # 使用亮度代替均值
            intensity = 0.299 * image[i, j, 2] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 0]
            source_weight = data_weight * (255 - intensity)**2 / 255**2
            sink_weight = data_weight * intensity**2 / 255**2
            graph.add_edge(source, idx, capacity=source_weight)
            graph.add_edge(idx, sink, capacity=sink_weight)

    # 添加平滑项
    for i in range(h):
        for j in range(w):
            idx = pixel_index(i, j)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4邻域
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    neighbor_idx = pixel_index(ni, nj)
                    color_diff = np.linalg.norm(image[i, j] - image[ni, nj])
                    spatial_dist = np.sqrt(di**2 + dj**2)
                    smooth_weight_val = smooth_weight * np.exp(-color_diff**2 / (2 * sigma_color**2)) \
                                        * np.exp(-spatial_dist**2 / (2 * sigma_space**2))
                    graph.add_edge(idx, neighbor_idx, capacity=smooth_weight_val)

    # 最大流计算
    _, flow_dict = nx.maximum_flow(graph, source, sink)

    # 根据分割结果更新图像
    denoised_image = np.zeros_like(image, dtype=np.uint8)
    reachable = nx.minimum_cut(graph, source, sink)[1][0]
    for i in range(h):
        for j in range(w):
            idx = pixel_index(i, j)
            if idx in reachable:
                denoised_image[i, j] = image[i, j]  # 保留前景
            else:
                denoised_image[i, j] = [0, 0, 0]  # 背景设为黑色

    return denoised_image


if __name__ == "__main__":
    # 加载彩色图像
    input_image_path = "gaussian_noise_colored.jpg"  # 替换为您的图像路径
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        raise FileNotFoundError("输入图像路径无效，请检查！")

    # 记录运行时间
    start_time = time.time()

    # 应用改进的彩色图像降噪
    denoised_image = image_denoising_bk_color_optimized(input_image)

    # 打印运行时间
    end_time = time.time()
    execution_time = end_time - start_time

    # 计算性能指标
    mse_value = mse(input_image, denoised_image)
    psnr_value = psnr(input_image, denoised_image)
    ssim_value = calculate_ssim_multichannel(input_image, denoised_image)
    epi_value = calculate_epi(input_image, denoised_image)

    # 打印性能指标
    print(f"MSE: {mse_value:.2f}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"EPI: {epi_value:.4f}")
    print(f"Execution Time: {execution_time:.4f} seconds")

    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()
