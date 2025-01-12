from PIL import Image
import numpy as np
import os

def apply_rayleigh_noise_color(image_array, scale=1.0):
    """
    给彩色图像添加瑞利噪声
    参数：
    - image_array: 彩色图像的三维像素值数组
    - scale: 噪声强度，值越大噪声越明显
    """
    # 转换为 numpy 数组
    image_array_np = np.array(image_array, dtype=float)

    # 生成瑞利噪声并添加到每个通道
    rayleigh_noise = np.random.rayleigh(scale, size=image_array_np.shape)
    noisy_image = image_array_np + rayleigh_noise

    # 裁剪到 [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image.astype(int)  # 返回整数类型数组

def image_to_array_color(image):
    """将彩色图像转换为三维像素值数组"""
    return np.array(image)

def array_to_image_color(image_array):
    """将三维像素值数组转换为彩色图像"""
    return Image.fromarray(image_array.astype(np.uint8))

def main():
    # 定义当前目录下的文件夹
    current_dir = os.getcwd()
    input_folder = os.path.join(current_dir)  # 输入文件夹
    output_folder = os.path.join(current_dir, "output_images")  # 输出文件夹

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图像文件
    file_list = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    scale_base = 20.0  # 基础噪声强度

    for index, file_name in enumerate(file_list):
        image_path = os.path.join(input_folder, file_name)
        input_image = Image.open(image_path).convert('RGB')  # 确保图像为彩色
        input_array = image_to_array_color(input_image)

        # 根据索引递增噪声强度
        scale = scale_base + index * 10.0

        # 添加瑞利噪声
        noisy_array = apply_rayleigh_noise_color(input_array, scale)

        # 保存加噪后的图像
        noisy_image = array_to_image_color(noisy_array)
        output_path = os.path.join(output_folder, f"{index + 1}.jpg")  # 使用数字顺序命名
        noisy_image.save(output_path)
        print(f"Processed {file_name} -> {index + 1}.jpg with scale {scale}")

if __name__ == "__main__":
    main()
