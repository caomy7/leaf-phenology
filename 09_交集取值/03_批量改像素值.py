import cv2
import numpy
from PIL import Image
import os
path = r"D:\06_scientific research\11_Image2ROI\05_AllROI_convert56\\"
save = r"D:\06_scientific research\11_Image2ROI\06_Allsites_roi01(OK)\\"
files = os.listdir(path)
for file in files:
    image = path +file
    im = Image.open(image)  # 打开图片
    print(im)
    pix = im.load()  # 导入像素
    width = im.size[0]  # 获取宽度
    height = im.size[1]  # 获取长度

    for x in range(width):
        for y in range(height):
            r, g, b = im.getpixel((x, y))
            rgb = (r, g, b)
            if (r == 255):
                im.putpixel((x, y), (0, 0, 0))
            else:
                im.putpixel((x, y), (1, 1, 1))

    im = im.convert('RGB')

    save_path = save + file
    print(save_path)
    im.save(save_path)



