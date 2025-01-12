from PIL import Image
import numpy as np

def apply_poisson_noise_color(image_array, scale=1.0):
    """
    给彩色图像添加泊松噪声
    参数：
    - image_array: 图像的三维像素值数组
    - scale: 噪声强度，默认值为1.0，值越大噪声越明显
    """
    # 将输入转换为 numpy 数组
    image_array_np = np.array(image_array, dtype=float)

    # 逐通道添加泊松噪声
    noisy_image = np.zeros_like(image_array_np)
    for c in range(3):  # 对 RGB 三个通道分别加噪
        noisy_channel = np.random.poisson(image_array_np[:, :, c] * scale) / scale
        noisy_image[:, :, c] = np.clip(noisy_channel, 0, 255)  # 限制范围

    return noisy_image.astype(np.uint8).tolist()

def image_to_array_color(image):
    """将彩色图像转换为三维像素值数组"""
    width, height = image.size
    return [[[image.getpixel((x, y))[c] for c in range(3)] for x in range(width)] for y in range(height)]

def array_to_image_color(image_array):
    """将三维像素值数组转换为彩色图像"""
    height, width, _ = len(image_array), len(image_array[0]), 3
    image = Image.new('RGB', (width, height))  # 彩色图像
    for y in range(height):
        for x in range(width):
            pixel = tuple(image_array[y][x])
            image.putpixel((x, y), pixel)
    return image

def main():
    # 打开输入图像
    image_path = "Lenna.jpg"
    input_image = Image.open(image_path).convert('RGB')  # 保持为彩色图像
    input_array = image_to_array_color(input_image)

    # 添加泊松噪声
    scale = 20.0  # 噪声强度
    noisy_array = apply_poisson_noise_color(input_array, scale)

    # 保存并显示加噪后的图像
    noisy_image = array_to_image_color(noisy_array)
    noisy_image.save('poisson_noise_colored.jpg')
    noisy_image.show()

if __name__ == "__main__":
    main()
