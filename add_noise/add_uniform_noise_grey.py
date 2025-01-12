from PIL import Image
import random

def add_uniform_noise(image_array, lower_bound=-20, upper_bound=20):
    """
    给图像添加均匀噪声
    参数可调：
    - lower_bound: 噪声的最小值
    - upper_bound: 噪声的最大值
    """
    height, width = len(image_array), len(image_array[0])
    noisy_image = [[0 for _ in range(width)] for _ in range(height)]

    for i in range(height):
        for j in range(width):
            noise = random.uniform(lower_bound, upper_bound)  # 生成均匀分布噪声
            noisy_value = image_array[i][j] + noise
            noisy_image[i][j] = max(0, min(255, int(noisy_value)))  # 保证像素值在[0, 255]范围内

    return noisy_image

def image_to_array(image):
    """将图像转换为二维像素值数组"""
    width, height = image.size
    return [[image.getpixel((x, y)) for x in range(width)] for y in range(height)]

def array_to_image(image_array):
    """将二维像素值数组转换为图像"""
    height, width = len(image_array), len(image_array[0])
    image = Image.new('L', (width, height))  # 灰度图
    for y in range(height):
        for x in range(width):
            image.putpixel((x, y), image_array[y][x])
    return image

def main():
    # 打开输入图像，转为灰度图
    image_path="Lenna.jpg"
    input_image = Image.open(image_path).convert('L')
    input_array = image_to_array(input_image)

    # 添加均匀噪声
    lower_bound = -20  # 噪声最小值
    upper_bound = 20   # 噪声最大值
    noisy_array = add_uniform_noise(input_array, lower_bound, upper_bound)

    # 保存并显示加噪后的图像
    noisy_image = array_to_image(noisy_array)
    noisy_image.save('uniform_noise_grey.jpg')
    noisy_image.show()

if __name__ == "__main__":
    main()
