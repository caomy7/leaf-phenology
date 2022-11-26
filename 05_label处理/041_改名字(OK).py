#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'

"""

import os

ROOT_DIR = os.path.abspath(r"E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\061111")
img_path = os.path.join(ROOT_DIR , r"E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\061111")
imglist = os.listdir(img_path)
# print(filelist)
# i = 0
for fileName in imglist:
    name = fileName[:-8]

    print(name)

    print(fileName)

    # os.rename(fileName , name)
    # i += 1
    if fileName.endswith('.txt.txt'):
        print(fileName)
        src = os.path.join(os.path.abspath(img_path) , fileName)  # 原先的图片名字
        dst = os.path.join(os.path.abspath(img_path) , name)  # 根据自己的需要重新命名,可以把'E_' + img改成你想要的名字
        os.rename(src , dst)  # 重命名,覆盖原先的名字