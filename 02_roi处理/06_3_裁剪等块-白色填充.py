from PIL import Image
import sys


def cut_image(image):
    width, height = image.size
    item_width = int(width / 4)
    box_list = []
    count = 0
    for j in range(0, 4):
        for i in range(0, 4):
            count += 1
            box = (i * item_width, j * item_width, (i + 1) * item_width, (j + 1) * item_width)
            box_list.append(box)
    print(count)
    image_list = [image.crop(box) for box in box_list]
    return image_list


def save_images(image_list):
    index = 1
    for image in image_list:
        image.save(r'D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\test_1' + str(index) + '.jpg')
        index += 1


if __name__ == '__main__':
    file_path = "001.jpg"
    # 打开图像
    image = Image.open(r'C:\Users\rockhuang\Desktop\colorspace\red3.png')
    # 将图像转为正方形，不够的地方补充为白色底色
    # image=fill_image(image)
    # 分为图像
    image_list = cut_image(image)
    # 保存图像
    save_images(image_list)