#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import os

path  = r'E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\167days\\'
files = os.listdir(path)
i=0
for file in files:
    text = open(path + file)
    i = i+1
    m = str(i)
    print(i)

    with open(file,'w+') as wf:
        wf.write('389 518 450 548 ')
        wf.write(m)