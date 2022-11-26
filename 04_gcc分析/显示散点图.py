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

import glob,os

path = r'E:\phenicam\roi_stats'
output = r"E:\phenicam\csv_output"
files = glob.glob(os.path.join(path, "*.csv"))

data_list = []
for f in files:
    # if f =="acadia_DB_1000_roistats.csv":

    Data = pd.read_csv(f)
    print(Data)
    x = Data['date']
    y = Data['gcc']
    print(x)

    # plt.scatter(x,y)
    # plt.show()