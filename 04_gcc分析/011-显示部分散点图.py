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
import pandas as pd
import operator
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
import glob,os

path = r'E:\phenicam\acadia_DB_1000_roistats.csv'
df = pd.read_csv(path)

x= df ['date']
y = df['gcc']

# print(a.index(min(a)))  # 返回第一个最小值的位置
# print(a.index(max(a)))  # 返回第一个最大值的位置
print(np.max(y))  # 返回最大值
print(np.min(y))  # 返回最小值
max_indx = np.argmax(y) # 返回第一个最小值的位置
min_indx = np.argmin(y)  # 返回第一个最大值的位置
# min_index,min_y = min(enumerate(y),key=operator.itemgetter)
# print(min_index,min_y)
plt.scatter(x,y,c="g")
plt.plot(max_indx,x[max_indx],'ks')
show_max='['+str(max_indx)+' '+str(x[max_indx])+']'
plt.annotate(show_max,xytext=(max_indx,x[max_indx]),xy=(max_indx,x[max_indx]))
plt.plot(min_indx,x[min_indx],'gs')
plt.show()