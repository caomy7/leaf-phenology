#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/05'

"""
import os
import re
import sys
path = r"E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\061111\\"
fileList = os.listdir(path)
currentpath = os.getcwd()
# print(fileList)
#
# num = 1
# 遍历文件夹中所有文件
for fileName in fileList:
    name = fileName[:-8]

    print(name)

    print(fileName)

    os.rename(fileName,name)

    # 改变编号，继续下一项
    # num = num + 1
print("***************************************")
os.chdir(currentpath)
# 刷新
sys.stdin.flush()

# print("修改后：" + str(os.listdir(r"E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\06222")))
