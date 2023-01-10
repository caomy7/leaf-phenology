import pandas as pd
from scipy import interpolate
import random
import datetime
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import numpy as np
import xml
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score




path2 = r"D:\06_scientific research\07_result\12_dukehw_SOS8daysEOS24cla"
birth_weight_file = r"D:\06_scientific research\07_result\12_dukehw_SOS8daysEOS24cla\results_SOS8EOS.xlsx"

data = pd.read_excel(birth_weight_file)  # 读取训练数据
print(data)

x=data["A"]
# y = data["B"]
y = data["C"]
# y = data["pre"]

rms = sqrt(mean_squared_error(x, y))
rmse = sqrt(mean_squared_error(x, y))
r2 = r2_score(x,y)
rms = round(rms,3)
rmse = round(rmse,3)
r2 = round(r2,3)
print(x,y)
print(x.shape,y.shape)
print(rms,rmse,r2)

fig =  plt.figure(figsize=(4,4))

plt.scatter(x,y,marker="o",color = "green",s=20,label = "RMSE")
plt.ylabel("Prediction")
plt.xlabel("True")
plt.title("Error")
plt.annotate(rms,xy=(4,20),c="blue")
plt.annotate(rmse,xy=(4,19),c="blue")
plt.annotate(r2,xy=(4,18),c="blue")


plt.text(0,20,"rms =",c="blue")
plt.text(0,19,"rmse=",c="blue")
plt.text(0,18,"r2 =",c="blue")
# plt.legend(loc = "best")
# plt.legend(loc = "upper right",fontsize =8)
# plt.legend(loc = "upper right")


y_smooth = scipy.signal.savgol_filter(y,59,3)

# plt.plot(x, y_smooth,linewidth = 2, color='r')
plt.savefig(r"D:\06_scientific research\07_result\12_dukehw_SOS8daysEOS24cla\Error_SOS8EOS24cla.tif")
plt.show()

