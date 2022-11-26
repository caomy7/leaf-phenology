#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/13'

"""
import cv2 as cv
import os
import time

"""
    转换tiff格式为png + 横向裁剪tiff遥感影像图
"""
def Convert_To_Png_AndCut(dir):
    files = os.listdir(dir)
    # ResultPath1 = r"F:\02_Data\phenocamV2\04_roi_labels\06_resize\roi_png"  # 定义转换格式png后的保存路径
    ResultPath2 = r"F:\02_Data\phenocamV2\04_roi_labels\06_resize\roi_resize\\"  # 定义裁剪后的保存路径

    for file in files:  # 这里可以去掉for循环
        a , b = os.path.splitext(file)  # 拆分影像图的文件名称
        this_dir = os.path.join(dir + "\\" + file)  # 构建保存 路径+文件名
        img = cv.imread(this_dir , 1)  # 读取tif影像
        # print(img)
        # 第二个参数是通道数和位深的参数，
        # IMREAD_UNCHANGED = -1  # 不进行转化，比如保存为了16位的图片，读取出来仍然为16位。
        # IMREAD_GRAYSCALE = 0  # 进行转化为灰度图，比如保存为了16位的图片，读取出来为8位，类型为CV_8UC1。
        # IMREAD_COLOR = 1   # 进行转化为RGB三通道图像，图像深度转为8位
        # IMREAD_ANYDEPTH = 2  # 保持图像深度不变，进行转化为灰度图。
        # IMREAD_ANYCOLOR = 4  # 若图像通道数小于等于3，则保持原通道数不变；若通道数大于3则只取取前三个通道。图像深度转为8位

        # cv.imwrite(ResultPath1 + a + ".png" , img)  # 保存为png格式
        # 下面开始裁剪-不需要裁剪tiff格式的可以直接注释掉
        hight = img.shape[0]  # opencv写法，获取宽和高
        width = img.shape[1]
        new_im = cv.resize(img , (224 , 224))
        cv.imwrite(ResultPath2 + a + b , new_im)
        end = time.clock()
        print("runing time:%s Seconds" % (end - start))




if __name__ == '__main__':
    start = time.clock()
    path = r"F:\02_Data\phenocamV2\04_roi_labels\roi"  # 遥感tiff影像所在路径
    # 裁剪影像图
    Convert_To_Png_AndCut(path)

