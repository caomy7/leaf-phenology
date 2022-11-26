#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/18'
"""

import os , sys
import glob
from PIL import Image

# %% refer to https://blog.csdn.net/xunan003/article/details/79052288

# VEDAI 图像存储位置
src_img_dir = r'E:\phenicam\02_gcc\07_images'
# VEDAI 图像的 ground truth 的 txt 文件存放位置
src_txt_dir = r"E:\phenicam\02_gcc\05_labels14242\\"
src_xml_dir = r"E:\phenicam\02_gcc\06_Anatations\\"

txt_list = os.listdir(src_txt_dir)
img_names = txt_list
# print(txt_list)

for img in img_names:
    img = img[:-4]
    print(img)
    im = Image.open((src_img_dir + '/' + img + '.jpg'))
    width , height = im.size

    # open the crospronding txt file
    gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
    # gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()

    # write in xml file
    # os.mknod(src_xml_dir + '/' + img + '.xml')
    xml_file = open((src_xml_dir + '/' + img + '.xml') , 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + str(img) + '.png' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    # write the region of image on xml file
    for img_each_label in gt:
        spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str(spt[4]) + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(spt[0]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(spt[1]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(spt[2]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(spt[3]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')

# 少了source 少了owner
