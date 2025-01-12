from PIL import Image
import numpy as np

def apply_gamma_noise_grey(image_array, shape=2.0, scale=20.0):
    """
    给灰度图像添加伽马噪声
    参数：
    - image_array: 图像的二维像素值数组
    - shape: 伽马分布的形状参数 (k)，控制噪声强度
    - scale: 伽马分布的尺度参数 (θ)，控制噪声范围
    """
    # 转换为 NumPy 数组
    image_array_np = np.array(image_array, dtype=np.float32)

    # 添加伽马噪声
    gamma_noise = np.random.gamma(shape, scale, size=image_array_np.shape)
    noisy_image = image_array_np + gamma_noise  # 添加噪声

    # 裁剪到 [0, 255] 并转换为整数
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

def image_to_array_grey(image):
    """将灰度图像转换为二维像素值数组"""
    width, height = image.size
    return [[image.getpixel((x, y)) for x in range(width)] for y in range(height)]

def array_to_image_grey(image_array):
    """将二维像素值数组转换为灰度图像"""
    height, width = len(image_array), len(image_array[0])
    image = Image.new('L', (width, height))  # 灰度图
    for y in range(height):
        for x in range(width):
            image.putpixel((x, y), int(image_array[y][x]))  # 确保像素值为整数
    return image

def main():
    # 打开输入图像
    image_path = "Lenna.jpg"
    input_image = Image.open(image_path).convert('L')  # 转为灰度图像
    input_array = image_to_array_grey(input_image)

    # 添加伽马噪声
    shape = 2.0  # 伽马分布的形状参数
    scale = 20.0  # 伽马分布的尺度参数
    noisy_array = apply_gamma_noise_grey(input_array, shape, scale)

    # 保存并显示加噪后的图像
    noisy_image = array_to_image_grey(noisy_array)
    noisy_image.save('gamma_noise_grey.jpg')
    noisy_image.show()

if __name__ == "__main__":
    main()
