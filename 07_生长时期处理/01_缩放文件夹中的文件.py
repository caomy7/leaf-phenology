#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/18'
# code is far away from bugs with the god animal protecting
"""
import os,sys
from PIL import ImageFile
import shutil,cv2
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
# root_dir = "E:/phenicam/02_gcc/02_original/03_56site/"
save = r"E:\phenicam\02_gcc\03_resize\01_image\\"

# def objFileName():
#     dirs = os.listdir(save)
#     return dirs

root_dir = r"E:\phenicam\02_gcc\02_original\03_56site"
list_dir = os.listdir(root_dir)
for dir in list_dir:
    files = os.listdir(root_dir +"\\"+ dir)
    print(files)

    file_name = save + str(dir)
    os.mkdir(file_name)
    print(file_name)

    for file in files:

        # print(file)
        filename = root_dir+"\\"+dir+"\\"+file
        # print(filename)
        # img = cv2.imread(filename)
        img = Image.open(filename)
        # print(img)
        # new_im = cv2.resize(img,(224 , 224))
        new_im = img.resize((224,224), Image.ANTIALIAS)
        print(new_im,file_name+"\\"+file)
        # shutil.copy(new_im, file)

        new_im.save(file_name+"\\"+file)  # 存储裁剪得到的图像
