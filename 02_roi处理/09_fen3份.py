import os, random, shutil
# -*- coding: UTF-8 -*-
import os
import random
import shutil


def eachFile(filepath):
    name_list = []
    pathDir = os.listdir(filepath)
    return pathDir


def divideTrainValiTest(source, dist):
    # pic_name = eachFile1(source)
    pic_name = eachFile(source)
    random.shuffle(pic_name)
    train_list = pic_name[0:int(0.7 * len(pic_name))]
    validation_list = pic_name[int(0.7 * len(pic_name)):int(0.9 * len(pic_name))]
    test_list = pic_name[int(0.9 * len(pic_name)):]
    for train_pic in train_list:
        # shutil.move(source + '/' + train_pic, dist + '/train/' + pos_or_neg + '/' + train_pic)
        shutil.move(source + '\\' + train_pic, dist + '\\train\\' + '\\' + train_pic)
    for validation_pic in validation_list:
        shutil.move(source + '/' + validation_pic, dist + '/val/' + validation_pic)

    for test_pic in test_list:
        shutil.move(source + '/' + test_pic, dist + '/test/' + test_pic)
    return


if __name__ == '__main__':
    # filepath = r'/your_path/raw_data'
    filepath = r'E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\01_image'
    # dist = r'/your_path/data'
    dist = r'E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\05_trainvaltest'
    divideTrainValiTest(filepath,dist)

