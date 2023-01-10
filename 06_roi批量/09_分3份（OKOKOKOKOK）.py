import os
import random
import shutil


def eachFile(source):
    # dir_names = os.listdir(source)
    # for names in dir_names:
    #     for i in ['train', 'val', 'test']:
    #         path = dist + '/' + i + '/' + names
    #         if not os.path.exists(path):
    #             os.makedirs(path)
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
        print(train)
        # print(dist)
        # path1 =os.listdir(dist +"\\"+ train.txt)
        # print(dist +"\\"+ train)
        with open("train.txt", 'a') as f:
            f.write(train+"\n")
        shutil.copy(source + '/' + train_pic, dist + '/train' + '/' + train_pic)
    for validation_pic in validation_list:
        val,extention = os.path.splitext(validation_pic)
        with open("val.txt", 'a') as f:
            f.write(val+"\n")
        shutil.copy(source + '/' + validation_pic, dist + '/val/' + '/' + validation_pic)

    for test_pic in test_list:
        test, extention = os.path.splitext(test_pic)
        with open("test.txt", 'a') as f:
            f.write(test+"\n")
        # with open(dist,'w+') as f:
        #     f.write(test)

        shutil.copy(source + '/' + test_pic, dist + '/test/' + '/' + test_pic)
    return


if __name__ == '__main__':
    # filepath = r'E:\IDM_phenocam\04_labels'  # 划分的源文件
    # dist = r'E:\IDM_phenocam\06_fen3fen'  # 生成的三个文件
    filepath = r'E:\IDM_phenocam\05_Anotations-均衡化'  # 划分的源文件
    dist = r'E:\IDM_phenocam\07_fen3fen均衡化'  # 生成的三个文件
##############注意：需要先命名train-val-test三个文件夹######################################

    divideTrainValiTest(filepath, dist)