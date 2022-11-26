#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/6/2'

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
import glob,os

path = r'E:\phenicam\acadia_DB_1000_roistats.csv'
files = pd.read_csv(path)
print(files)
x = files['date']
y = files['gcc']

# print(x,y)

Data = pd.DataFrame(x,y)
# print(Data)
# df2=Data[Data['date']== 2007-11]
# Data['date=2017']
# print(df2)
# m=df2['date']
# n =df2['gcc']

Data['date'] = pd.to_datetime(Data['date'])

df = Data.set_index('date')
plt.scatter(x,y,c="red")
print(type(df))
print(df.index)
print(type(df.index))

print(df.shape)
# print(df['2007'].head(2))

plt.xlabel("date")
plt.ylabel("gcc")


# plt.show()