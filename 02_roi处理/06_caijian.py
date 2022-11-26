# coding: utf-8
from PIL import Image
import os
import os.path
import numpy as np
import cv2
import random

# 指明被遍历的文件夹
rootdir = r'D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\test'
for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
    for filename in filenames:
        print('parent is :' + parent)
        print('filename is :' + filename)
        currentPath = os.path.join(parent, filename)
        print('the fulll name of the file is :' + currentPath)

        img = Image.open(currentPath)
        print(img.format, img.size, img.mode)
        # img.show()
        img_size = img.size
        m = img_size[0]  # 读取图片的宽度
        n = img_size[1]  # 读取图片的高度
        w = 224 # 设置你要裁剪的小图的宽度
        h = 224  # 设置你要裁剪的小图的高度
        for i in range(100):  # 裁剪为100张随机的小图
            x = random.randint(0, m - w)  # 裁剪起点的x坐标范围
            y = random.randint(0, n - h)  # 裁剪起点的y坐标范围
            region = img.crop((x, y, x + w, y + h))  # 裁剪区域
        # box1 = (0, 0, 224, 224)  # 设置左、上、右、下的像素
        # image1 = img.crop(box1)  # 图像裁剪


        region.save(r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\test_1" + '\\' + filename)  # 存储裁剪得到的图像