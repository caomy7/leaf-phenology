import csv
import pandas as pd
import matplotlib.pyplot as plt

path = r"D:\06_scientific research\07_result\02_roi_pic\00 result_pic\result_csv (复制 1)\val_scores.csv"

data = pd.read_csv(path)  # 读取训练数据
# data = pd.read_csv(path,header=None)  # 读取训练数据
# data = csv.reader(open(path,"r"))
print(data)

df_plot = data.plot(x= "epochs")
plt.ylabel("Value")
plt.xlabel("Epochs")
plt.title("Train result")
plt.legend(loc = "best")

plt.savefig(r"D:\06_scientific research\07_result\02_roi_pic\1_TrainResult.tif")
plt.show()

