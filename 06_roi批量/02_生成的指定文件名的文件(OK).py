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
path1 = r"E:\phenicam\01_ROI_56\01_56site\bitterootvalley"
path2 = r"E:\phenicam\01_ROI_56\04_ROILabels\bitterootvalley"

def objFileName():
    dirs = os.listdir(path1)
    return dirs

def copy_img():
    local_img_name=r"E:\phenicam\01_ROI_56\04_ROILabels\bitterootvalley\0.tif"
    #指定要复制的图片
    # print(objFileName)
    for i in objFileName():
        # print(i)
        old_name,extension = os.path.splitext(i)
        # print(old_name)

        new_obj_name = old_name+".jpg"
        # new_obj_name = i
        # print(new_obj_name)
        # print(local_img_name)
        print(path2 + "\\" + new_obj_name)
        shutil.copy(local_img_name, path2 + "\\"+new_obj_name)
    os.remove(local_img_name)

if __name__ == '__main__':
    copy_img()
    # objFileName()