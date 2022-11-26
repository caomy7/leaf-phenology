#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/05'
# code is far away from bugs with the god animal protecting

"""
import os
import test


path = r"E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\061_text\\"
files = os.listdir(path)
# print(files)
file = 1
# for file in files:
file2 = r"E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\062_text"


for file in files:
    # print(file)
    # with open(file , "r" , encoding="utf-8") as f1 , open("%s.bak" % file , "w" , encoding="utf-8") as f2:

    df = open(path+file, 'r', encoding='utf-8')
    print(df)
    # print(df.read)

    lines = df.readlines()
    for line in lines:
        line = line.replace('roi','999',)
        # print(line)
        print(line)
    # file2.write(line)
