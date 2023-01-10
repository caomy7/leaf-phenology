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
    train_list = pic_name[0:int(0.8 * len(pic_name))]
    validation_list = pic_name[int(0.8 * len(pic_name)):int(0.9 * len(pic_name))]
    test_list = pic_name[int(0.9 * len(pic_name)):]
    for train_pic in train_list:
        train,extention = os.path.splitext(train_pic)
        # print(train)
        # print(dist)
        path1 =os.listdir(dist +"\\"+ train)
        # path1 =os.listdir(dist +"\\"+ train)
        print(path1)
        # with open(path1, 'w+') as f:
        #     f.write(train)
        # shutil.move(source + '/' + train_pic, dist + '/train/' + pos_or_neg + '/' + train_pic)
        # shutil.copy(source + '\\' + train_pic, dist + '\\train' + '\\' + train_pic)
    for validation_pic in validation_list:
        val,extention = os.path.splitext(validation_pic)
        # shutil.copy(source + '/' + validation_pic, dist + '/val/' + '/' + validation_pic)

    for test_pic in test_list:
        test, extention = os.path.splitext(test_pic)
        # with open(dist,'w+') as f:
        #     f.write(test)

        # shutil.copy(source + '/' + test_pic, dist + '/test/' + '/' + test_pic)
    return


if __name__ == '__main__':
    filepath = r'E:\IDM_phenocam\05_Anotations-均衡化'  # 划分的源文件
    dist = r'E:\IDM_phenocam\07_fen3fen均衡化'  # 生成的三个文件

    divideTrainValiTest(filepath, dist)