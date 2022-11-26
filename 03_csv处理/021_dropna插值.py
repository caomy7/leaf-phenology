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

path = r"E:\acadia\acadia_DB_1000_roistats.csv"
records = pd.read_csv(path,skiprows=17)
print(records)

x = records['date']
y = records['gcc']
# print(x,y)

data = [x,y]
# print(data)
Data=pd.DataFrame(data).T#数据有三列，列名分别为one,two,three
df = Data.fillna(method='backfill' , limit=5)
result =df.interpolate(method="linear")
print(result)


result.to_csv("test2.csv",index=False, header=True)
