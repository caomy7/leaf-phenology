#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/12'

"""
import os , sys

path1 = r'E:\phenicam\01_ROI——56\01_56site\\'  # 指定名称文件夹所在路径
path2 = r'E:\phenicam\01_ROI——56\04_ROILabels\roi_files\\'  # 新建文件夹所在路径


def MkDir():
    dirs = os.listdir(path1)
    print(dirs)
    # dirs = ['']

    for dir in dirs:
        print(dir)
        file_name = path2 + str(dir)
        # print(dir)
        print(file_name)
        os.mkdir(file_name)


MkDir()