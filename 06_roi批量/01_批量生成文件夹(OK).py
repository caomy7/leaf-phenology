#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/12'

"""
import os , sys

path1 = r'D:\06_scientific research\05_gee_labels\04_45site_resize\01_image\\'  # 指定名称文件夹所在路径
path2 = r'D:\06_scientific research\05_gee_labels\04_45site_resize\各站点list_name\生成的文件夹名\\'  # 新建文件夹所在路径


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