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

birth_weight_file = r"D:\06_scientific research\01_train_roi\02_roistas_csv\01_roistats\harvardbarn2_DB_1000_roistats.csv"
birth_data = []
with open(birth_weight_file) as csvfile:
    # csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    csv_reader = pd.read_csv(csvfile)  # 使用csv.reader读取csvfile中的文件
    # print(csv_reader)
data = csv_reader
# wp = data.drop_duplicates(['date'])
wp = data.groupby(['date']).gcc.transform(max)
wp = data[data.gcc == wp]
print(wp)

X = wp["date"]
Y = wp["gcc"]
# X_float = int(X)
# plt.figure(figsize=(5,3),dpi=600) # figsize设置图片大小，dpi设置清晰度
# xticks=list(range(0,368,40)) # 这里设置的是x轴点的位置（40设置的就是间隔了）
x = range(0,368)
# x = np.linspace(0, 369,1).reshape(369,1)

# print(X.shape,Y.shape,x.shape)

# plt.axhline(y=0,ls=":",c="yellow")#添加水平直线
plt.axvline(x=126,ls="--",c="blue",label = "SOS")#添加垂直直线
plt.axvline(x=291,ls='--',c="blue",label = "EOS")#添加垂直直线;'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
plt.text(100,0.42,"SOS",fontsize = 12,c="blue")
plt.text(293,0.42,"EOS",fontsize = 12,c="blue")



plt.scatter(x,Y,marker="o",color = "green",s=10,label = "value of Gcc")
# plt.scatter(X,Y,marker="o",color = "green",s=2,label = "value of Gcc")


my_x_ticks = np.arange(0,400, 50)
plt.xticks(my_x_ticks)
# x_major_locator=MultipleLocator(300)
# plt.xticks(X, calendar.month_name[1:13],color='blue',rotation=60)  #参数x空值X轴的间隔，第二个参数控制每个间隔显示的文本，后面两个参数控制标签的颜色和旋转角度
# plt.plot(x,Y)
plt.xlim(0,366)

plt.ylabel("Green Chromatic Coordinate(Gcc)")
plt.xlabel("Day Of Year")
plt.title("The value of Gcc in harvardbarn2")
plt.legend(loc = "best")

# f1 = np.polyfit(x, Y, 3)
# y2 = parameter[0] * x ** 3 + parameter[1] * x ** 2 + parameter[2] * x + parameter[3]
# p1 = np.poly1d(f1)
# plt.scatter(x, Y)
# print(x.shape,Y.shape)

# p1 = polyfit(x, Y,3)

# plt.show()
# regr = linear_model.LinearRegression()
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=30))
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# regr.fit(x,Y)
# plt.plot(x,regr.predict(x),color = "blue",linewidth = 3)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')   #设置 上、右 两条边框不显示

# def f_fit(x,a,b):
#     return a*np.sin(x)+b
# def f_show(x,p_fit):
#     a,b=p_fit.tolist()
#     return a*np.sin(x)+b
# def f(x):
#     return 2*np.sin(x)+8
#
# p_fit,pcov=curve_fit(f_fit,x,Y)#曲线拟合
# print(p_fit)#最优参数
# print(pcov)#最优参数的协方差估计矩阵
# y1=f_show(x,p_fit)
################多项式拟合#############
# order=8
# c=np.polyfit(x,Y,order)#拟合多项式的系数存储在数组c中
# yy=np.polyval(c,x)#根据多项式求函数值
# f_liner=np.polyval(c,x)
# f_liner=np.polyval(c,x)

# plt.plot(x, f_liner, color='black')
########################SG滤波############
# def create_x(size, rank):
#     x = []
#     for i in range(2 * size + 1):
#         m = i - size
#         row = [m**j for j in range(rank)]
#         x.append(row)
#     x = np.mat(x)
#     return x
# def sgolayfilt(data, window_size, rank):
#     m = (window_size - 1) / 2
#     odata = data[:]
#     # 处理边缘数据，首尾增加m个首尾项
#     for i in range(m):
#         odata.insert(0,odata[0])
#         odata.insert(len(odata),odata[len(odata)-1])
#     # 创建X矩阵
#     x = create_x(m, rank)
#     # 计算加权系数矩阵B
#     b = (x * (x.T * x).I) * x.T
#     a0 = b[m]
#     a0 = a0.T
#     # 计算平滑修正后的值
#     ndata = []
#     for i in range(len(data)):
#         y = [odata[i + j] for j in range(window_size)]
#         y1 = np.mat(y) * a0
#         y1 = float(y1)
#         ndata.append(y1)
#     return ndata

y_smooth = scipy.signal.savgol_filter(Y,47,3)
# y_smooth = scipy.signal.savgol_filter(Y,43,3)

plt.plot(x, y_smooth, color='r')


plt.savefig(r"D:\06_scientific research\07_result\02_GccScatter\harvardbarn2.png")
plt.show()
