#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/05'
#
"""

import os

def alter(file,old_str,new_str):
    with open(file, "r", encoding="utf-8") as f1,open("%s.txt"% file, "w+", encoding="utf-8") as f2:
        print(f1)
        for lin in f1:
            print(lin)
            if old_str in lin:
                lin = lin.replace(old_str, new_str)
            f2.write(lin)
            print(f2)
        os.remove(file)
        os.rename("%s.txt" % file, file)




# path = r"E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\061_text\acadia_2016_09_19_120005.txt"
path = r"E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\061_text\acadia_2016_06_04_110001.txt"
# path = r"E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\061_text\acadia_2016_09_23_130001.txt"

# path = r"E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\061_text\acadia_2016_10_04_120002.txt"
# path = r"E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\061_text\acadia_2016_06_14_120001.txt"
folder_path, file_name = os.path.split(path)
print(folder_path)
print(file_name)
i=file_name[15:17]
# i=file_name[16:17]
print(i)
s = str(int(i)+24)
print(s)
alter(path,"roi",s)




