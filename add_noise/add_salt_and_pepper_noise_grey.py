from PIL import Image
import random

def add_salt_and_pepper_noise(image_array, salt_prob=0.02, pepper_prob=0.02):
    """
    给图像添加椒盐噪声
    参数可调：
    - salt_prob: 盐噪声的概率（白点）
    - pepper_prob: 椒噪声的概率（黑点）
    """
    height, width = len(image_array), len(image_array[0])
    noisy_image = [[image_array[i][j] for j in range(width)] for i in range(height)]

    for i in range(height):
        for j in range(width):
            rand = random.random()
            if rand < salt_prob:
                noisy_image[i][j] = 255 
            elif rand < salt_prob + pepper_prob:
                noisy_image[i][j] = 0    

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
    image_path="Lenna.jpg"
    input_image = Image.open(image_path).convert('L')
    input_array = image_to_array(input_image)

    # 添加椒盐噪声
    salt_prob = 0.02   # 盐噪声概率,值越大，白点越多
    pepper_prob = 0.02 # 椒噪声概率，值越大，黑点越多
    noisy_array = add_salt_and_pepper_noise(input_array, salt_prob, pepper_prob)

    noisy_image = array_to_image(noisy_array)
    noisy_image.save('salt_and_pepper_noise_grey.jpg')
    noisy_image.show()

if __name__ == "__main__":
    main()
