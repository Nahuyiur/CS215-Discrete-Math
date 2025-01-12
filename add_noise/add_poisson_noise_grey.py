from PIL import Image
import numpy as np

def apply_poisson_noise(image_array, scale=1.0):
    """
    给灰度图像添加泊松噪声
    参数：
    - image_array: 图像的二维像素值数组
    - scale: 噪声强度，默认值为1.0，值越大噪声越明显
    """
    # 将列表转换为 numpy 数组
    image_array_np = np.array(image_array, dtype=float)

    # 生成泊松噪声并限制像素值范围
    noisy_image = np.random.poisson(image_array_np * scale) / scale
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    # 转回列表形式以保持一致
    return noisy_image.tolist()

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
            image.putpixel((x, y), image_array[y][x])
    return image

def main():
    # 打开输入图像
    image_path = "Lenna.jpg"
    input_image = Image.open(image_path).convert('L')  # 转为灰度图像
    input_array = image_to_array_grey(input_image)

    # 添加泊松噪声
    scale = 20.0  # 噪声强度，值越大噪声越明显
    noisy_array = apply_poisson_noise(input_array, scale)

    # 保存并显示加噪后的图像
    noisy_image = array_to_image_grey(noisy_array)
    noisy_image.save('poisson_noise_grey.jpg')
    noisy_image.show()

if __name__ == "__main__":
    main()
