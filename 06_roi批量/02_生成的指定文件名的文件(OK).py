#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/12'
"""

import os,sys
import shutil
#这个库复制文件比较省事
path1 = r"E:\IDM_phenocam\02_resize_select"
path2 = r"E:\IDM_phenocam\03_txt16716"

def objFileName():
    dirs = os.listdir(path1)
    return dirs

def copy_img():
    local_img_name=r"E:\IDM_phenocam\dukehw_2016_01_01_113106.txt"
    #指定要复制的图片
    # print(objFileName)
    for i in objFileName():
        # print(i)
        old_name,extension = os.path.splitext(i)
        # print(old_name)

        new_obj_name = old_name+".txt"
        # new_obj_name = i
        # print(new_obj_name)
        # print(local_img_name)
        print(path2 + "\\" + new_obj_name)
        # if local_img_name == i:
        shutil.copy(local_img_name, path2 + "\\"+new_obj_name)
    # os.remove(local_img_name)

if __name__ == '__main__':
    copy_img()
    # objFileName()