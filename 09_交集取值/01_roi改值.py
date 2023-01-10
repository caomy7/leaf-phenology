import cv2
import numpy as np
from PIL import Image
import os
path = r"D:\06_scientific research\11_Image2ROI\01_dukehw_extract_Roi\dukehw_2016_03_25_120109.jpg"

im = Image.open(path)  # 打开图片
pix = im.load()  # 导入像素
width = im.size[0]  # 获取宽度
height = im.size[1]  # 获取长度

for x in range(width):
    for y in range(height):
        r, g, b = im.getpixel((x, y))
        rgb = (r, g, b)
        if (r >= 1):
            im.putpixel((x, y), (0, 0, 0))
        else:
            im.putpixel((x, y), (1, 1, 1))

im = im.convert('RGB')
im.save(r"D:\06_scientific research\11_Image2ROI\01_dukehw_extract_Roi\dukehw333_roi.jpg")



