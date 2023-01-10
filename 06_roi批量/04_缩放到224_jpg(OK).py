#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/13'
"""

from PIL import Image
# from libtiff import TIFF
import os
import os.path
import numpy as np
import cv2
import random

# rootdir = r'F:\02_Data\phenocamV2\04_roi_labels\roi'
# tiff文件解析成图像序列：读取tiff图像

    #     print(img)
    #     w,h =img.size
    #     # print(w,h)
    #     scale_w = w/224
    #     scale_h = h/224
    #     # new_im = img.resize((224, 224), Image.ANTIALIAS)
    #     # new_im = img.resize((224, 224))
    #     new_im = img.resize((int(w / scale_w) , int(h / scale_h)))
    #     print(new_im)
    #
    # # new_im.save(r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\train3_resize" + '\\' + filename)  # 存储裁剪得到的图像
    #

import cv2
import tifffile
import time
import numpy as np
import time
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

rootdir =  r'E:\IDM_phenocam\dukehw'
# out_txt_path = '../out_word_all.box'
# out_tiff_path = '../out_word_all.tif'

for parent , dirnames , filenames in os.walk(rootdir):  # 遍历每一张图片
    for filename in filenames:
        currentPath = os.path.join(parent , filename)
        img = Image.open(currentPath)
        print(img)

        # gray_img = cv2.imread(currentPath , cv2.IMREAD_GRAYSCALE)
        # new_gray = gray_img[np.newaxis , ::]
        # print('gray_img shape:{}, new_gray shape:{}'.format(gray_img.shape , new_gray.shape))
        # global cnt_num
        # w , h = new_gray.size
        new_im = img.resize((224,224), Image.ANTIALIAS)
        print(new_im)
        new_im.save(r"E:\IDM_phenocam\01_resizedukehw" + '\\' + filename)  # 存储裁剪得到的图像



