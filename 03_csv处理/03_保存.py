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
import csv


path = r"E:\acadia\acadia_DB_1000_1day.csv"
row = np.arange(0,24)
csv = pd.read_csv(path,encoding='utf-8', skiprows=row,na_values="NA")
# print(csv)

x = csv['date']
y = csv['gcc_mean']
# print(x,y)

data = [x,y]

# test=pd.DataFrame(columns=data,data=data)#数据有三列，列名分别为one,two,three
df=pd.DataFrame(data=data)#数据有三列，列名分别为one,two,three
print(df)
df2 = df.stack()
result = df2.unstack(0)
print(result)

result.to_excel('testcsv.xls',encoding='gbk')
