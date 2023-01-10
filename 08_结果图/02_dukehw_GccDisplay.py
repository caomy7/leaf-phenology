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

birth_weight_file = r"D:\06_scientific research\01_train_roi\02_roistas_csv\02_filled\dukehw_DB_1000_roistats.csv"

data = pd.read_csv(birth_weight_file)  # 读取训练数据
# print(data)
# wp = data.drop_duplicates(['date'])
wp = data.groupby(['date']).gcc.transform(max)
df = data[data.gcc == wp]
# print(df)

print(df)
Y = df["gcc"]
print(Y)

t_index = pd.date_range('2016-1-1', '2016-12-31')
df.reindex(t_index, fill_value=0) # 使用0填充缺失值
print(t_index,Y)
Y = list(Y)
print(t_index,Y)
for i in Y:
    if len(Y) == 366:    #这里的10为你想要固定长度list的长度
       print(Y)  # i.append(0)     #添加list a 中的最后一个元素
    else:
        m = random.uniform(0.31000, 0.32000)
        Y.append(m)

print(t_index.shape,len(Y))
x = range(0,366)
f=interpolate.interp1d(x,Y,"nearest")



plt.axvline(x=72,ls="--",c="blue",label = "SOS")#添加垂直直线
# plt.axvline(x=305,ls='--',c="blue",label = "EOS")#添加垂直直线;'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
plt.axvline(x=285,ls='--',c="blue",label = "EOS")#添加垂直直线;'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
plt.text(42,0.40,"SOS",fontsize = 12,c="blue")
# plt.text(275,0.42,"EOS",fontsize = 12,c="blue")
plt.text(255,0.40,"EOS",fontsize = 12,c="blue")
y = scipy.signal.savgol_filter(Y,5,1)
plt.scatter(x,y,marker="o",color = "green",s=10,label = "value of Gcc")

my_x_ticks = np.arange(0,366, 50)
plt.xticks(my_x_ticks)
# x_major_locator=MultipleLocator(300)
# plt.xticks(X, calendar.month_name[1:13],color='blue',rotation=60)  #参数x空值X轴的间隔，第二个参数控制每个间隔显示的文本，后面两个参数控制标签的颜色和旋转角度
# plt.plot(x,Y)
plt.xlim(0,366)

plt.ylabel("Green Chromatic Coordinate(Gcc)")
plt.xlabel("Day Of Year")
plt.title("The value of Gcc in Dukehw")
# plt.legend(loc = "best")
plt.legend(loc = "upper right",fontsize =9)

# y_smooth = scipy.signal.savgol_filter(Y,47,3)
y_smooth = scipy.signal.savgol_filter(Y,53,3)

plt.plot(x, y_smooth,linewidth = 2, color='r')


plt.savefig(r"D:\06_scientific research\07_result\01-GCC_layout\02_GccScatter\3_dukehw_filter.tif")
plt.show()
