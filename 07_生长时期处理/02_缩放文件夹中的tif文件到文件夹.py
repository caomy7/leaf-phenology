#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/18'
"""
import cv2 as cv
import os
import time


def Convert_To_Png_AndCut(dir):
    files = os.listdir(path + dir)
    save = r"E:\phenicam\02_gcc\03_resize\02_roi\\"  # 定义裁剪后的保存路径

    file_name = save + str(dir)
    os.mkdir(file_name)
    # print(file_name)

    for file in files:  # 这里可以去掉for循环
        a , b = os.path.splitext(file)  # 拆分影像图的文件名称
        this_dir = os.path.join(path + dir + "\\" + file)  # 构建保存 路径+文件名
        print(this_dir)
        img = cv.imread(this_dir , 1)  # 读取tif影像
        # print(img)


        new_im = cv.resize(img,(224 , 224))
        cv.imwrite(file_name +"\\"+ a + b , new_im)
        # new_im.save(file_name + "\\" + file)

        # end = time.clock()
        # print("runing time:%s Seconds" % (end - start))

if __name__ == '__main__':
    # start = time.clock()
    path = r"E:\phenicam\02_gcc\02_original\04_ROILabels\\"  # 遥感tiff影像所在路径
    # 裁剪影像图
    list_dir = os.listdir(path)
    for dir in list_dir:
        Convert_To_Png_AndCut(dir)