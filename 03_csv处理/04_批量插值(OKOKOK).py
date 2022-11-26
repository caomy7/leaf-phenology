#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/6/1'
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import glob,os

path = r'E:\phenicam\roi_stats'
output = r"E:\phenicam\csv_output"
files = glob.glob(os.path.join(path, "*.csv"))
# print(file)
# files =  os.listdir(path)
# files_csv = list(filter(lambda x:x[-4:]=='.csv', files))
data_list = []
for f in files:
    domain1 = os.path.abspath(path)  # 待处理文件位置
    info = os.path.join(domain1, f)  # 拼接出待处理文件名字
    domain2 = os.path.abspath(output)  # 处理完文件保存地址
    outfo = os.path.join(domain2,f)  # 拼接出新文件名字
    print(info , "开始处理")
    records = pd.read_csv(f, skiprows=17)
    # print(records)
    x = records['date']
    y = records['gcc']
    # print(x,y)
    ######################################转换###################
    data = [x , y]
    # print(data)
    Data = pd.DataFrame(data).T  # 数据有三列，列名分别为one,two,three
    print(Data)
    df = Data.fillna(method='backfill' , limit=5)
    result = df.interpolate(method="nearest")

    print(result)
    # ------省略数据处理过程----------------------
    result.to_csv(outfo,encoding='utf-8')  # 将数据写入新的csv文件
    print(info,"处理完")


#
# # fig = plt.figure(2)
# fig = plt.subplot(2,2,1)
# # plt.scatter(x,y)
# # plt.plot(x,yy)
#
# #