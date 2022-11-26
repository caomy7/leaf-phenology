import os
import shutil
import glob
import cv2

path = r'D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_train_roi\01train\roi\archboldbahia_phenocam_data_20200523080415\phenocamdata\archboldbahia\2017\03'
new_path = r'D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_train_roi\01train\03_trainData\4个站点\label\label3'
for root, dirs, files in os.walk(path):  # 提取文件夹下所有jpg文件复制转移到新的文件夹
    for i in range(len(files)):
        if files[i][-3:] == 'jpg' or files[i][-3:] == 'JPG':
            file_path = root + '/' + files[i]
            new_file_path = new_path + '/' + files[i]
            print(new_path,new_file_path)
            shutil.copy(file_path, new_file_path)

img_path = glob.glob('D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_train_roi\01train\extract/*.jpg')  # 获取新文件夹下所有图片
i = 1
for each in img_path:
    img = cv2.imread(each, cv2.IMREAD_UNCHANGED)
    cv2.imshow('Image', img)  # 顺次显示每一帧
    print(img)
    k = cv2.waitKey(0)  # 每一帧等待时间为无穷大
    if k == ord('s'):  # 当按s键时保存此帧，按其他键则不保存而跳到下一帧
        cv2.imwrite('D:/want/%d.jpg' % i, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
    i = i + 1