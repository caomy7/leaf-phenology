import pandas as pd
from scipy import interpolate
import random
import datetime
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import numpy as np

path2 = r"D:\06_scientific research\07_result\01-GCC_layout\02_GccScatter"
birth_weight_file = r"D:\06_scientific research\01_train_roi\02_roistas_csv\02_filled\dukehw_DB_1000_roistats.csv"

data = pd.read_csv(birth_weight_file)  # 读取训练数据
print(data)
# print(data.shape)
# df = data.drop_duplicates(['date'])
wp = data.groupby(['date']).gcc.transform(max)
df = data[data.gcc == wp]

print(df,df.shape)
Date =df["date"]
Gcc = df["gcc"]
################### 新建数据框 ########################
d = pd.DataFrame()
d['date'] = Date

d['date'] = pd.to_datetime(d['date'])
d['val'] = Gcc
helper = pd.DataFrame({'date': pd.date_range(d['date'].min(), d['date'].max())})
d = pd.merge(d, helper, on='date', how='outer').sort_values('date')
# print(d)
# print(d.shape)

############## 异常值处理 #################
# u = d["val"].mean()  # 计算均值
# std = d["val"].std()  # 计算标准差
# error = d[np.abs(d["val"] - u) > 1.39*std]
# data_c = d[np.abs(d["val"] - u) <= 1.39*std]
# print('异常值共%i条' % len(error))
# # print(data_c)
# d["gcc"] =data_c
# # print(type(data_c))
# # print(d)
# print(type(d["gcc"]))
############################ 对Gcc进行插值 ################
# # d['val'] = d['val'].interpolate(method='linear')
# # d['val'] = data_c.interpolate(method='linear')
# # print(d)
# # print(d.shape)
# X = d['date']
# print(X)


############################ Display Plot ############################
Y = d['val']
# Y = list(Y)
print(Y)
X = range(-1,366)
print(type(X),type(Y))
y = scipy.signal.savgol_filter(Y,3,1)
plt.scatter(X,y)

# text = open(path2 + "txt.txt", "w+")
# text.write(str(d))
# my_x_ticks = np.arange(1,366, 50)
# plt.xticks(my_x_ticks)
# plt.xlim(0,366)
plt.ylabel("Green Chromatic Coordinate(Gcc)")
plt.xlabel("Day Of Year")
plt.title("The value of Gcc in Dukehw")
plt.legend(loc = "best")
# plt.legend(loc = "upper right",fontsize =9)
y_smooth = scipy.signal.savgol_filter(Y,25,1)

plt.plot(X, y_smooth,linewidth = 2, color='r')
plt.show()


