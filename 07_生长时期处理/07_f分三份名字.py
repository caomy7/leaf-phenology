#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/18'
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ASUS'
__mtime__ = '2020/06/13'
"""
import os, random, shutil,time
def make_dir(source, target):
    '''
    创建和源文件相似的文件路径函数
    :param source: 源文件位置
    :param target: 目标文件位置
    '''
    dir_names = os.listdir(source)
    for names in dir_names:
        # print(names)
        for i in ['train', 'val', 'test']:
            path = target + '/' + i + '/' + names
            if not os.path.exists(path):
                os.makedirs(path)

def divideTrainValiTest(source, target):
    '''
        创建和源文件相似的文件路径
        :param source: 源文件位置
        :param target: 目标文件位置
    '''
    # 得到源文件下的种类
    pic_name = os.listdir(source)
    # print(pic_name)
    # 对于每一类里的数据进行操作
    for classes in pic_name:
        print(classes)
        # 得到这一种类的图片的名字
        pic_classes_name = os.listdir(os.path.join(source, classes))
        random.shuffle(pic_classes_name)
        # 按照8：1：1比例划分
        train_list = pic_classes_name[0:int(0.8 * len(pic_classes_name))]
        val_list = pic_classes_name[int(0.8 * len(pic_classes_name)):int(0.9 * len(pic_classes_name))]
        test_list = pic_classes_name[int(0.9 * len(pic_classes_name)):]

        # 对于每个图片，移入到对应的文件夹里面
        for train_pic in train_list:
            shutil.copyfile(source + '/' + classes + '/' + train_pic, target + '/train/' + classes + '/' + train_pic)
        for validation_pic in val_list:
            shutil.copyfile(source + '/' + classes + '/' + validation_pic,target + '/val/' + classes + '/' + validation_pic)
        for test_pic in test_list:
            shutil.copyfile(source + '/' + classes + '/' + test_pic, target + '/test/' + classes + '/' + test_pic)
# end = time.clock()
if __name__ == '__main__':
    filepath = r'F:\02_Data\phenocamV2\04_roi_labels\06_resize\train_resize'#划分的源文件
    dist = r'F:\02_Data\phenocamV2\04_roi_labels\07_resize2datasets'#生成的三个文件
    make_dir(filepath, dist)
    divideTrainValiTest(filepath, dist)
    # start = time.clock()
    # print("runing time %"%(end - start))