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
import csv

path = r"E:\acadia\acadia_DB_1000_1day.csv"
row = np.arange(0,24)
csv = pd.read_csv(path,encoding='utf-8', skiprows=row,na_values= ['male'])

file = csv.to_excel('aca.xlsx', sheet_name='data',encoding='utf-8')
print(file)

record = pd.read_excel('aca.xlsx')
x = record['date']
y = record['gcc_mean']
print(x,y)


# data =([x],[y])
# data =[x,y]


# name=['data','gcc']
# test=pd.DataFrame(columns=data,data=data)#数据有三列，列名分别为one,two,three
# test=pd.DataFrame(data=data)#数据有三列，列名分别为one,two,three
# print(test)
# test.to_csv('testcsv.csv',encoding='gbk')
# test.to_csv('testcsv.xls',encoding='gbk')


# with open("aca.csv","w+") as f:
#     f.close()
# with open("aca.xls","w+") as f:

    # f_csv = csv.writer(f)
    #
    # f_csv.writecolumn(x)
    #
    # f_csv.writecolumn(y)
    # write(x,y)
# fig = plt.figure(2)
plt.plot(x,y)
plt.show()