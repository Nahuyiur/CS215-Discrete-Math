from PIL import Image
import numpy as np

def apply_gamma_noise_color(image_array, shape=2.0, scale=20.0):
    """
    给彩色图像添加伽马噪声
    参数：
    - image_array: 图像的三维像素值数组
    - shape: 伽马分布的形状参数 (k)，控制噪声强度
    - scale: 伽马分布的尺度参数 (θ)，控制噪声范围
    """
    height, width, channels = len(image_array), len(image_array[0]), len(image_array[0][0])
    noisy_image = np.zeros((height, width, channels), dtype=np.uint8)

    for c in range(channels):  # 对每个通道单独添加噪声
        channel_array = np.array([[image_array[y][x][c] for x in range(width)] for y in range(height)], dtype=np.float32)
        gamma_noise = np.random.gamma(shape, scale, size=channel_array.shape)
        noisy_channel = channel_array + gamma_noise
        noisy_image[:, :, c] = np.clip(noisy_channel, 0, 255).astype(np.uint8)

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
    input_image = Image.open(image_path).convert('RGB')  # 确保图像为彩色
    input_array = image_to_array_color(input_image)

    # 添加伽马噪声
    shape = 2.0  # 伽马分布的形状参数
    scale = 20.0  # 伽马分布的尺度参数
    noisy_array = apply_gamma_noise_color(input_array, shape, scale)

    # 保存并显示加噪后的图像
    noisy_image = array_to_image_color(noisy_array)
    noisy_image.save('gamma_noise_colored.jpg')
    noisy_image.show()

if __name__ == "__main__":
    main()
