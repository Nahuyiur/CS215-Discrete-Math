from PIL import Image
import numpy as np

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
    width, height = image.size
    return [[[image.getpixel((x, y))[c] for c in range(3)] for x in range(width)] for y in range(height)]

def array_to_image_color(image_array):
    """将三维像素值数组转换为彩色图像"""
    height, width, _ = len(image_array), len(image_array[0]), 3
    image = Image.new('RGB', (width, height))  # 彩色图像
    for y in range(height):
        for x in range(width):
            pixel = tuple(int(image_array[y][x][c]) for c in range(3))  # 转为整数
            image.putpixel((x, y), pixel)
    return image

def main():
    # 打开输入图像
    image_path = "Lenna.jpg"
    input_image = Image.open(image_path).convert('RGB')  # 确保图像为彩色
    input_array = image_to_array_color(input_image)

    # 添加瑞利噪声
    scale = 30.0  # 噪声强度，值越大噪声越明显
    noisy_array = apply_rayleigh_noise_color(input_array, scale)

    # 保存并显示加噪后的图像
    noisy_image = array_to_image_color(noisy_array)
    noisy_image.save('rayleigh_noise_colored.jpg')
    noisy_image.show()

if __name__ == "__main__":
    main()
