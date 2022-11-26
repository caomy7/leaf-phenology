#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/5/31'
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
# print(records)
x = records['date'].values.tolist()
y = records['gcc'].values.tolist()
# print(x,y)
data = [x,y]
print(data)
df=pd.DataFrame(data=data,)#数据有三列，列名分别为one,two,three
# print(df)
df2 = df.stack()
Data = df2.unstack(0)
# print(result)
# print(x.type,y.type)

# result['date'] = pd.to_datetime(result['date'])

# helper = pd.DataFrame({'date': pd.date_range(result['date'].min(), result['date'].max())})
# d = pd.merge(result,re helper, on='date', how='outer').sort_values('date')
result = Data.fillna(method='backfill' , limit=5)
# r = result.interp1d(method='linear')
# f = interp1d(x,y)
# print(f)
# print(x.size,y.size)
# x_new = np.arange(int(x))
# x =str(x)
# print(x)
# x_new = np.arange(0,4024)
# x_new = np.arange(0,4024,1)
# print(x_new)
# print(x_new.size)
# print(y)
# y_new = f(y)
# print(y_new.size)
# print(x,y_new)


# f = interp1d(x,y,kind="linear")
# y_new =f(x)
# print(x,y)
# print(result)

# fig = plt.figure(2)
fig = plt.subplot(2,2,1)
plt.plot(x,y)
# plt.plot(x,yy)

# plt.show()
r.to_excel('testcsv.xls',encoding='gbk')




# data = [x,yy]
# df=pd.DataFrame(data=data)#数据有三列，列名分别为one,two,three
# print(df)
# df2 = df.stack()
# result = df2.unstack(0)
# print(result)
#
# result.to_excel('testcsv.xls',encoding='gbk')