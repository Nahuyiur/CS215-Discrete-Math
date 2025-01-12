from PIL import Image
import random
import math

def add_gaussian_noise_color(image_array, mean=0, std_dev=20):
    """
    给彩色图像添加高斯噪声
    参数：
    - mean: 噪声的均值
    - std_dev: 噪声的标准差
    """
    height, width, channels = len(image_array), len(image_array[0]), len(image_array[0][0])
    noisy_image = [[[0 for _ in range(channels)] for _ in range(width)] for _ in range(height)]

    for i in range(height):
        for j in range(width):
            for c in range(channels):  # 对每个通道单独加噪
                noise = random.gauss(mean, std_dev)  # 生成高斯噪声
                noisy_value = image_array[i][j][c] + noise
                noisy_image[i][j][c] = max(0, min(255, int(noisy_value)))  # 保证像素值在[0, 255]范围内

    return noisy_image

def image_to_array_color(image):
    """将彩色图像转换为三维像素值数组"""
    width, height = image.size
    return [[[image.getpixel((x, y))[c] for c in range(3)] for x in range(width)] for y in range(height)]

def array_to_image_color(image_array):
    """将三维像素值数组转换为彩色图像"""
    height, width, channels = len(image_array), len(image_array[0]), 3
    image = Image.new('RGB', (width, height))  # 彩色图像
    for y in range(height):
        for x in range(width):
            pixel = tuple(image_array[y][x][c] for c in range(channels))
            image.putpixel((x, y), pixel)
    return image

def main():
    # 打开输入图像
    image_path = "Lenna.jpg"
    input_image = Image.open(image_path)  # 确保输入图像是RGB格式
    input_array = image_to_array_color(input_image)

    # 添加高斯噪声
    mean = 0      # 噪声均值
    std_dev = 30  # 噪声标准差
    noisy_array = add_gaussian_noise_color(input_array, mean, std_dev)

    # 保存并显示加噪后的图像
    noisy_image = array_to_image_color(noisy_array)
    noisy_image.save("gaussian_noise_colored.jpg")
    noisy_image.show()

if __name__ == "__main__":
    main()
