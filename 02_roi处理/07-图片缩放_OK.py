# coding: utf-8
from PIL import Image
from libtiff import TIFF
import os
import os.path
import numpy as np
import cv2
import random

# 指明被遍历的文件夹

rootdir = r'F:\02_Data\phenocamV2\04_roi_labels\roi'
for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
    for filename in filenames:
        print('parent is :' + parent)
        print('filename is :' + filename)
        currentPath = os.path.join(parent, filename)
        print('the fulll name of the file is :' + currentPath)


        img = Image.open(currentPath)


        # from libtiff import TIFF
        #
        # tif = TIFF.open('currentPath', mode='r')
        # img = tif.read_image()

        # img = cv2.imread("currentPath")

        # img = TIFF.open(currentPath)



        print(img.format, img.size, img.mode)
        # img.show()
        # img_size = img.size
        # m = img_size[0]  # 读取图片的宽度
        #
        # n = img_size[1]  # 读取图片的高度
        w, h = img.size

        scale_w = 1296 / 224
        scale_h = 976 / 220
        # new_im = img.resize((int(w / scale_w), int(h / scale_h)), Image.ANTIALIAS)
        # new_im = img.resize((224,224), Image.ANTIALIAS)
        # new_im = cv2.resize(img,(int(w / scale_w), int(h / scale_h)))
        # new_im = cv2.resize(img,(224,224))
        # print(new_im)
        new_im = img.resize((int(w / scale_w), int(h / scale_h)))
        print(new_im)
        # cv.imwrite(save_dir + '\\' + filename, new_im)

        # new_im.save('new-' + filename)
        # new_im.close()
        # new_im.save(r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\train3_resize" + '\\' + filename)  # 存储裁剪得到的图像

