#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/18'

"""
import os,sys
# root_dir = r"E:\phenicam\02_gcc\04_使用的52个resize数据\01_image"
# root_dir = r"E:\phenicam\02_gcc\04_52site_resize\01_image"
root_dir = r"E:\phenicam\02_gcc\04_52site_resize\02_roi"
list_dir = os.listdir(root_dir)
for i,dir in enumerate(list_dir):
    files = os.listdir(root_dir +"\\"+ dir)
    # file = os.path.splitext(dir)
    # print(file)
    # print(dir)
    for i,file in enumerate(files):
        filename = root_dir + "\\" + dir + "\\" + file
        # file_name = os.path.splitext(filename)
        # print(filename)
        # print(file_name)
    print(i+1)