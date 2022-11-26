#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/19'

"""
import os,sys
# root_dir = r"E:\phenicam\02_gcc\04_使用的52个resize数据\01_image"

root_dir = r"E:\phenicam\02_gcc\05_labels14242\\"
list_dir = os.listdir(root_dir)
for i,dir in enumerate(list_dir):
    # print(dir)
    file = os.path.join(root_dir + dir)
    with open(file,"r") as f:
        content = f.read()
        # print(content)
