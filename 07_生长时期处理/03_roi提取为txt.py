#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/18'
"""

import cv2
import numpy as np
import os
# from osgeo import gdal
# root_dir = r'E:\phenicam\02_gcc\04_52site_resize\02_roi\acadia\\'
# root_dir = r'E:\phenicam\02_gcc\04_52site_resize\02_roi\drippingsprings\\'
# root_dir = r'E:\phenicam\02_gcc\04_52site_resize\02_roi\coweeta\\'
# root_dir = r'E:\phenicam\02_gcc\04_52site_resize\02_roi\harvardlph\\'
# root_dir = r'E:\phenicam\02_gcc\04_52site_resize\02_roi\howland2\\'
root_dir = r'E:\phenicam\02_gcc\04_52site_resize\02_roi\shalehillsczo\\'
# root_dir = r'E:\phenicam\02_gcc\04_52site_resize\02_roi\umichbiological2\\'

txt_dir = r'E:\phenicam\02_gcc\07_3labels\\'
images = os.listdir(root_dir)
for i,image in enumerate(images):

    print(image)
    img = cv2.imread(root_dir+image)
    # cv2.imshow('img',img)
    cv2.waitKey(0)

    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    ret , binary = cv2.threshold(gray,0, 255 , cv2.THRESH_BINARY)
    # contours , hier = cv2.findContours(binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    contours , hier = cv2.findContours(binary, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    # print(contours)
    cnts = contours[1]
    # cv2.drawContours(img , cnt , -1 , (0 , 0 , 255) , 3)  #绘制轮廓，img输入原图，contours已经找出的多个轮廓，color颜色，thickness粗细（小于0为填充）
    # cv2.imshow("img" , img)
    cv2.waitKey(0)


    height , width = img.shape[:2]

    boxes = []
    if contours:
        txt = open(txt_dir + image[:-4] + '.txt' , 'w')
        for contour in cnts:
        # for i,contour in enumerate(cnts):
            x , y , w , h = cv2.boundingRect(contour)  # get top-left(X,Y) and height and width

            # top-left: tl  bottom-right:tr
            expand = 0
            xmin = max(0 , x - expand)
            ymin = max(0 , y - expand)
            xmax = min(width , x + w + expand)
            ymax = min(height , y + h + expand)

            dw = 1 / width
            dh = 1 / height
            x_new = (xmin + xmax) / 2 * dw
            y_new = (ymin + ymax) / 2 * dh
            w_new = (xmax - xmin) / 2 * dw
            h_new = (ymax - ymin) / 2 * dh

            # boxes.append([xmin,ymin,xmax,ymax])
            boxes.append([x_new , y_new , w_new ,h_new])

            ID = 0
            # if i <108 or i >270:
            if i <96 or i >217:
                # m =0
                s = ID

                # print(i,s)
            else:
                s =i-96
                # print(i,s)
            print(s)
            # print(i)
            print(s,x_new,y_new,w_new,h_new)
        txt.write('{} {} {} {} {}\n'.format(s,x_new,y_new,w_new,h_new))

        # print(boxes)
        txt.close()