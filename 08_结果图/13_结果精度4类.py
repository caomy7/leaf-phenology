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

birth_weight_file = r"D:\06_scientific research\07_result\06_dukehw4classes\dukehw4cla.xlsx"

data = pd.read_excel(birth_weight_file)  # 读取训练数据
print(data)

x=data["x"]
# y = data["B"]
y = data["pre"]
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
plt.annotate(rms,xy=(0.7,2.4),c="blue")
plt.annotate(rmse,xy=(0.7,2.6),c="blue")
plt.annotate(r2,xy=(0.7,2.8),c="blue")


plt.text(0.2,2.4,"rms =",c="blue")
plt.text(0.2,2.6,"rmse=",c="blue")
plt.text(0.2,2.8,"r2 =",c="blue")
# plt.legend(loc = "best")
# plt.legend(loc = "upper right",fontsize =8)
# plt.legend(loc = "upper right")


y_smooth = scipy.signal.savgol_filter(y,59,3)

# plt.plot(x, y_smooth,linewidth = 2, color='r')
plt.savefig(r"D:\06_scientific research\07_result\06_dukehw4classes\Error_30days_4cla.tif")
plt.show()

