from PIL import Image
import random

def add_salt_and_pepper_noise_color(image_array, salt_prob=0.01, pepper_prob=0.01):
    """
    给彩色图像添加椒盐噪声。
    参数：
    - salt_prob: 添加盐噪声（白点）的概率。
    - pepper_prob: 添加椒噪声（黑点）的概率。
    """
    height, width, channels = len(image_array), len(image_array[0]), 3
    noisy_image = [[[0 for _ in range(channels)] for _ in range(width)] for _ in range(height)]

    for i in range(height):
        for j in range(width):
            if random.random() < salt_prob:
                noisy_image[i][j] = [255, 255, 255]  # 盐噪声：白点
            elif random.random() < pepper_prob:
                noisy_image[i][j] = [0, 0, 0]  # 椒噪声：黑点
            else:
                noisy_image[i][j] = image_array[i][j]  # 保留原始像素值

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

    # 添加椒盐噪声
    salt_prob = 0.02  # 盐噪声概率
    pepper_prob = 0.02  # 椒噪声概率
    noisy_array = add_salt_and_pepper_noise_color(input_array, salt_prob, pepper_prob)

    # 保存并显示加噪后的图像
    noisy_image = array_to_image_color(noisy_array)
    noisy_image.save('salt_and_pepper_noise_colored.jpg')
    noisy_image.show()

if __name__ == "__main__":
    main()
