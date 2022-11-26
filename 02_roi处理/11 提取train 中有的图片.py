#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/04'
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/03'
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
import operator
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
import glob,os
import cv2
import os

def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        # print(filename)  # 仅仅是为了测试
        img = cv2.imread(directory_name + "/" + filename)
        # label = cv2.imread(path + "/" + filename)
        #####显示图片#######
        # cv2.imshow(filename, img)
        cv2.waitKey(0)
        #####################
        a = str(filename)
        name = "1000008.jpg","1000010.jpg","1000011.jpg","1000021.jpg","1000041.jpg","1000045.jpg","1000066.jpg","1000069.jpg","1000070.jpg","1000073.jpg","1000077.jpg","1000096.jpg","1000110.jpg","1000113.jpg","1000120.jpg","1000121.jpg","1000132.jpg","1000135.jpg","1000148.jpg","1000156.jpg","1000163.jpg","1000178.jpg","1000204.jpg","1000208.jpg","1000222.jpg","1000231.jpg","1000244.jpg","1000245.jpg","1000251.jpg","1000252.jpg","1000267.jpg","1000293.jpg","1000295.jpg","1000325.jpg","1000333.jpg","1000337.jpg","1000338.jpg","1000345.jpg","1000347.jpg","1000359.jpg","1000377.jpg","1000380.jpg","1000385.jpg","1000409.jpg","1000411.jpg","1000420.jpg","1000429.jpg","1000430.jpg","1000443.jpg","1000448.jpg","1000464.jpg","1000489.jpg","1000491.jpg","1000497.jpg","1000509.jpg","1000512.jpg","1000524.jpg","1000531.jpg","1000565.jpg","1000572.jpg","1000584.jpg","1000613.jpg","1000614.jpg","1000615.jpg","1000647.jpg","1000670.jpg","1000675.jpg","1000677.jpg","1000679.jpg","1000680.jpg","1000701.jpg","1000717.jpg","1000723.jpg","1000747.jpg","1000749.jpg","1000750.jpg","1000760.jpg","1000773.jpg","1000787.jpg","1000790.jpg","1000800.jpg","1000802.jpg","1000815.jpg","1000819.jpg","1000844.jpg","1000862.jpg","1000863.jpg","1000878.jpg","1000886.jpg","1000892.jpg","1000899.jpg","1000901.jpg","1000905.jpg","1000907.jpg","1000936.jpg","1000963.jpg","1000976.jpg","1000980.jpg","1000991.jpg","1000995.jpg","1001004.jpg","1001007.jpg","1001017.jpg","1001021.jpg","1001036.jpg","1001046.jpg","1001058.jpg"

        if a in name:

            print("save img...")

            # cv2.imwrite(r"E:\phenicam\01_acadia\acadia_DB_1000\01_image" + "/" + filename,img)
            cv2.imwrite(r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\test\labels" + "/" + filename,img)
read_directory(r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\labels")#这里传入所要读取文件夹的绝对路径，加引号（引号不能省略！）
