from PIL import Image

def add_gaussian_noise(image_array, mean=0, std=25):
    """
    给图像添加高斯噪声
    参数可调：
    - mean: 噪声的均值
    - std: 噪声的标准差
    """
    height, width = len(image_array), len(image_array[0])
    noisy_image = [[0 for _ in range(width)] for _ in range(height)]

    for i in range(height):
        for j in range(width):
            noise = random_gaussian(mean, std)
            noisy_pixel = image_array[i][j] + noise
            noisy_image[i][j] = max(0, min(255, int(noisy_pixel)))  # 限制像素值在[0, 255]范围内

    return noisy_image

def random_gaussian(mean, std):
    """使用 Box-Muller 方法生成高斯分布随机数"""
    import math
    import random
    u1 = random.random()
    u2 = random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z0 * std + mean

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
    image_path="Lenna.jpg"
    input_image = Image.open(image_path).convert('L')
    input_array = image_to_array(input_image)

    # 添加高斯噪声
    mean = 0  
    std = 30   
    noisy_array = add_gaussian_noise(input_array, mean, std)

    noisy_image = array_to_image(noisy_array)
    noisy_image.save('gaussian_noise_grey.jpg')
    noisy_image.show()

if __name__ == "__main__":
    main()
