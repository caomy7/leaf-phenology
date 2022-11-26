#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/07/22'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏ ┓   ┏ ┓
            ┏┛ ┻━━━┛ ┻━┓
            ┃    ☃     ┃
            ┃  ┳┛  ┗┳  ┃
            ┃     ┻    ┃
            ┗━┓      ┏━┛
              ┃      ┗━━━┓
              ┃神兽保佑   ┣┓
              ┃永无BUG！  ┏┛
              ┗┓┓┏━┳┓┏┛
               ┃┫┫ ┃┫┫
               ┗┻┛ ┗┻┛
"""
import os
import cv2
import re

root_dir = r"E:\phenicam\02_gcc\05_labels_data"
# target_txt = r"E:\phenicam\02_gcc\05_labels_noresize\\"

images = os.listdir(root_dir)
print(images)
for image in images:
    img = root_dir+"\\"+image
    # img = cv2.imread(root_dir+image)
    # print(img)

    with open(img, "r") as f:
        lines = f.readline()
        line = lines.split(" ",5)
        # print(line)
        l0 = line[0]
        l1 = line[1]
        l2 = line[2]
        l3 = line[3]
        l4 = line[4]
        print(l0,l1,l2,l3,l4)
        m1 = l1*112
        m2 = l2*112
        m3 = l3*112
        m4 = l4*112
        print(m1,m2,m3,m4)
        m = line[l0,m1,m2,m3,m4]
        print(m)
        f.write(re.sub(line,m))


