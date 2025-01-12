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
    if mse_value == 0:  
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))


def calculate_ssim(image1, image2):
    """计算单通道图像的 SSIM"""
    return ssim(image1, image2, data_range=image2.max() - image2.min())


def calculate_epi(original, filtered):
    """计算增强像素差异 (EPI)"""
    diff = np.abs(original - filtered)
    return np.mean(diff)


def image_denoising_bk_grayscale(image, data_weight=500, smooth_weight=50, sigma_color=50, sigma_space=10):
    """
    图像降噪 - 基于最大流-最小割算法 (NetworkX)
    :param image: 输入图像 (灰度图)
    :param data_weight: 数据项权重
    :param smooth_weight: 平滑项权重
    :param sigma_color: 像素颜色相似性参数
    :param sigma_space: 像素空间相似性参数
    :return: 降噪后的图像
    """
    h, w = image.shape
    num_pixels = h * w
    source = num_pixels  # Source node index
    sink = num_pixels + 1  # Sink node index

    graph = nx.DiGraph()

    def pixel_index(x, y):
        return x * w + y
    
    image = image.astype(np.float64)

    # 添加数据项（源点和汇点）
    start_time = time.time()
    for i in range(h):
        for j in range(w):
            idx = pixel_index(i, j)
            intensity = np.clip(image[i, j], 0, 255)
            source_weight = data_weight * (255 - intensity)**2 / 255**2
            sink_weight = data_weight * intensity**2 / 255**2
            graph.add_edge(source, idx, capacity=source_weight)
            graph.add_edge(idx, sink, capacity=sink_weight)

    # 添加平滑项（像素相邻关系）
    start_time = time.time()
    for i in range(h):
        for j in range(w):
            idx = pixel_index(i, j)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]: 
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    neighbor_idx = pixel_index(ni, nj)
                    color_diff = min(abs(image[i, j] - image[ni, nj]), 50)
                    spatial_dist = np.sqrt(di**2 + dj**2)
                    smooth_weight_val = smooth_weight * np.exp(-color_diff**2 / (2 * sigma_color**2)) \
                                        * np.exp(-spatial_dist**2 / (2 * sigma_space**2))
                    graph.add_edge(idx, neighbor_idx, capacity=smooth_weight_val)

    # 最大流计算
    start_time = time.time()
    _, flow_dict = nx.maximum_flow(graph, source, sink)

    # 根据流量结果更新图像
    denoised_image = np.zeros_like(image, dtype=np.uint8)
    reachable = nx.minimum_cut(graph, source, sink)[1][0]  
    for i in range(h):
        for j in range(w):
            idx = pixel_index(i, j)
            denoised_image[i, j] = 255 if idx in reachable else 0

    return denoised_image


if __name__ == "__main__":
    input_image_path = "gamma_noise_grey.jpg" 
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise FileNotFoundError("输入图像路径无效，请检查！")

    # 记录总运行时间
    total_start_time = time.time()

    # 图像降噪
    denoised_image = image_denoising_bk_grayscale(input_image)

    # 打印总运行时间
    execution_time = time.time() - total_start_time

    # 计算性能指标
    mse_value = mse(input_image, denoised_image)
    psnr_value = psnr(input_image, denoised_image)
    ssim_value = calculate_ssim(input_image, denoised_image)
    epi_value = calculate_epi(input_image, denoised_image)

    print(f"MSE: {mse_value:.2f}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"EPI: {epi_value:.4f}")
    print(f"Execution Time: {execution_time:.4f} seconds")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(input_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_image, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
