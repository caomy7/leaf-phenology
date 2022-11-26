import cv2
import tifffile
import time
import numpy as np
import time
import os

img_path = r'D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\test_labels'
# out_txt_path = '../out_word_all.box'
out_tiff_path = r'D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\test_labels_2.jpg'

tiff_list = None

# with open(out_txt_path, 'wb') as f:
dir_list = os.listdir(img_path)
cnt_num = 0

for dir_name in dir_list:
    dir_path = os.path.join(img_path, dir_name)
    img_list = os.listdir(dir_path)
    pwd = os.getcwd()
    os.chdir(dir_path)

    for img in img_list:

        print('dir_path:{}'.format(dir_path))
        gray_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        new_gray = gray_img[np.newaxis, ::]
        print('gray_img shape:{}, new_gray shape:{}'.format(gray_img.shape, new_gray.shape))
        # global cnt_num
        if cnt_num == 0:
            print('cnt_num == 0')
            tiff_list = new_gray
        else:
            print('np.append')
            tiff_list = np.append(tiff_list, new_gray, axis=0)
            print('tiff_list shape:{}'.format(tiff_list.shape))

        content = '{} 2 2 60 60 {}\n'.format(dir_name, cnt_num)
        print(content)
        f.write(content.encode('UTF-8'))
        cnt_num += 1
    os.chdir(pwd)

tifffile.imsave(out_tiff_path, tiff_list)

print('tiff_list shape:{}'.format(tiff_list.shape))