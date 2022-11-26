#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/18'
"""
import cv2
import os
import shutil
import time
# save = r"F:\02_Data\phenocamV2\04_roi_labels\train"
save = r"E:\phenicam\02_gcc\07_images"
# path =r"E:\phenicam\01_ROI_56\01_56site"
path =r"E:\phenicam\02_gcc\04_45site_resize\01_image"

files = os.listdir(path)
# print(files)
for file in files:
    # print(file)
    sub_file = os.listdir(path +"\\"+ file)
    # print(sub_file)
    for i in sub_file:
        # print(path+"\\"+file+"\\"+i)
        # print(save+"\\"+i)
        shutil.copy(path+"\\"+file+"\\"+i,save+"\\"+i)