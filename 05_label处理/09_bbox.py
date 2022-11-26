#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
# from osgeo import gdal

root_dir = r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\脚本\05_label处理"
images = os.listdir(root_dir)
images = ['acadia_DB_2000_02.tif']
# images = ['26.tif','27.tif']
print(images)
print(root_dir)
img = cv2.imread(root_dir+"//"+images)
cv2.imshow("img",img)
cv2.waitKey(0)