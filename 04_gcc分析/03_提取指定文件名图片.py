#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/03'
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
# import pandas as pd
import operator
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
import glob, os
import cv2
import os


# path = r"E:\phenicam\01_acadia\acadia_DB_1000\02_csv\GT_2016.xlsx"
path = r"D:\06_scientific research\05_gee_labels\06_one_site\02_dukehw\06_fen3fen\train.txt"
save =r"D:\04_Study\02_pytorch_github\10_回归\03_模型搭建\02_Regression\data\Dukehw\train"
def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        print(filename)  # 仅仅是为了测试
        img = cv2.imread(directory_name + "/" + filename)
        #####显示图片#######
        # cv2.imshow(filename, img)
        cv2.waitKey(0)
        #####################
        a = str(filename[:-4])
        print(a)
        if a in path:
            print("save img...")
            cv2.imwrite( save + "//" + filename, img)


read_directory(r"D:\04_Study\02_pytorch_github\10_回归\02_PhenocamDataset\01_dukehw656\03_Images656")  # 这里传入所要读取文件夹的绝对路径，加引号（引号不能省略！）
# read_directory(r"E:\phenicam\03_caryinstitute\2016\02")#这里传入所要读取文件夹的绝对路径，加引号（引号不能省略！）
