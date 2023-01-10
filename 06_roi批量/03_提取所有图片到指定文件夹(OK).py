#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/12'
"""

import cv2
import os
import shutil
import time
# save = r"F:\02_Data\phenocamV2\04_roi_labels\train"
save = r"D:\06_scientific research\10_回归\03_45phenocamRGB"
# path =r"E:\phenicam\01_ROI_56\01_56site"
path =r"D:\06_scientific research\10_回归\02_45siteRGB"
start = time.clock()
files = os.listdir(path)
# print(files)
for file in files:
    # print(file)
    sub_file = os.listdir(path +"\\"+ file)
    print(sub_file)
    for i in sub_file:
        print(path+"\\"+file+"\\"+i)
        print(save+"\\"+i)
        shutil.copy(path+"\\"+file+"\\"+i,save+"\\"+i)
end = time.clock()
print("finish")
print("runing time:%s Seconds"%(end-start))