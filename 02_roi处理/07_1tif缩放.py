from libtiff import TIFF
from scipy import misc
from PIL import Image
from PIL import Image
import os
import os.path
import cv2
import numpy as np
from libtiff import TIFFfile
#
# ##tiff文件解析成图像序列
# ##tiff_image_name: tiff文件名；
# ##out_folder：保存图像序列的文件夹
# ##out_type：保存图像的类型，如.jpg、.png、.bmp等
# def tiff_to_image_array(tiff_image_name, out_folder, out_type):
#     tif = TIFF.open(tiff_image_name, mode = "r")
#     idx = 0
#     for im in list(tif.iter_images()):
#         #
#         im_name = out_folder + str(idx) + out_type
#         misc.imsave(im_name, im)
#         print(im_name, 'successfully saved!!!')
#         idx = idx + 1
#     return
# ##图像序列保存成tiff文件
# ##image_dir：图像序列所在文件夹
# ##file_name：要保存的tiff文件名
# ##image_type:图像序列的类型
# ##image_num:要保存的图像数目
#
#
#
def image_array_to_tiff(image_dir, file_name, image_type, image_num):
    out_tiff = TIFF.open(file_name, mode = 'w')
    #这里假定图像名按序号排列
    for i in range(0, image_num):
        image_name = image_dir + str(i) + image_type
        # image_array = Image.open(image_name)
        image_array = cv2.open(image_name)
        #缩放成统一尺寸
        img = image_array.resize((224, 224), Image.ANTIALIAS)
        out_tiff.write_image(img, compression = None, write_rgb = True)
    out_tiff.close()
    return




#tiff文件解析成图像序列：读取tiff图像
def tiff_to_read(tiff_image_name):
 # tif = TIFF.open(tiff_image_name, mode = "r")
 tif = TIFFflile(tiff_image_name)
 im_stack = list()
 for im in list(tif.iter_images()):
  im_stack.append(im)
 return
 #根据文档,应该是这样实现,但测试中不管是tif.read_image还是tif.iter_images读入的矩阵数值都有问题

#图像序列保存成tiff文件：保存tiff图像
# def write_to_tiff(tiff_image_name, im_array, image_num):
#  tif = TIFF.open(tiff_image_name, mode = 'w')
#  for i in range(0, image_num):
#   im = Image.fromarray(im_array[i])
#   #缩放成统一尺寸
#   im = im.resize((224, 224), Image.ANTIALIAS)
#   tif.write_image(im, compression = None)
#  out_tiff.close()
#  return





# 指明被遍历的文件夹
rootdir = r'D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\val_labels'
# image_array_to_tiff(rootdir,)
out_folder = r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\val_labels_2"  # 存储裁剪得到的图像
for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
    for filename in filenames:
        currentPath = os.path.join(parent, filename)
        img = cv2.imread(currentPath)
        # image_count = np.sum(currentPath)
        # print(image_count)
        image_type = currentPath.format(currentPath)
        print(image_type)
        tiff_to_read(currentPath)
        # tiff_to_image_array(currentPath, out_folder, out_type="jpg")
        image_array_to_tiff(rootdir,currentPath,image_type='tif',image_num=366)

# return currentPath
# img = Image.open(currentPath)
# img = cv2.imread(currentPath)
# print(img)
#