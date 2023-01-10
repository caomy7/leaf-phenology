import xml
import csv
import scipy
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.spatial
import numpy as np
import pandas as pd
from scipy import signal
from numpy import polyfit, poly1d
from matplotlib.pyplot import MultipleLocator
import calendar
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from scipy import interpolate
import datetime
from scipy.optimize import leastsq
import random

birth_weight_file = r"D:\06_scientific research\01_train_roi\02_roistas_csv\01_roistats\proctor_DB_1000_roistats.csv"

data = pd.read_csv(birth_weight_file)  # 读取训练数据
# print(data)
# wp = data.drop_duplicates(['date'])
wp = data.groupby(['date']).gcc.transform(max)
df = data[data.gcc == wp]
# print(df)

# print(df)
Y = df["gcc"]
print(Y.shape)


t_index = pd.date_range('2016-1-1', '2016-12-31')
df.reindex(t_index, fill_value=0) # 使用0填充缺失值
print(t_index,Y)
Y = list(Y)
# print(t_index,Y)
# for i in Y:
#     if len(Y) == 366:    #这里的10为你想要固定长度list的长度
#        print(Y)  # i.append(0)     #添加list a 中的最后一个元素
#     else:
#         m = random.uniform(0.31000, 0.32000)
#         Y.append(m)
#
# # print(t_index.shape,len(Y))
# x = range(0,366)
# f=interpolate.interp1d(x,Y,"nearest")
#     # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
# y=f(x)

x = range(0,361)

plt.axvline(x=131,ls="--",c="blue",label = "SOS")#添加垂直直线
plt.axvline(x=292,ls='--',c="blue",label = "EOS")#添加垂直直线;'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
plt.text(100,0.365,"SOS",fontsize = 12,c="blue")
plt.text(295,0.365,"EOS",fontsize = 12,c="blue")
plt.scatter(x,Y,marker="o",color = "green",s=10,label = "value of Gcc")

my_x_ticks = np.arange(0,366, 50)
plt.xticks(my_x_ticks)

plt.xlim(0,366)

plt.ylabel("Green Chromatic Coordinate(Gcc)")
plt.xlabel("Day Of Year")
plt.title("The value of Gcc in Protoc")
plt.legend(loc = "best")
# plt.legend(loc = "upper right",fontsize =8)

y_smooth = scipy.signal.savgol_filter(Y,53,3)

plt.plot(x, y_smooth, color='r')


plt.savefig(r"D:\06_scientific research\07_result\02_GccScatter\Protoc.png")
plt.show()
