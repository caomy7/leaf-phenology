from PIL import Image
import os

######## 需要裁剪的图片位置#########
rootdir = r'D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\test0'


path_img = './GAN_img/'

img_dir = os.listdir(rootdir)
print(img_dir)

'''
（左上角坐标(x,y)，右下角坐标（x+w，y+h）
'''

for i in range(len(img_dir)):
    #####根据图片名称提取id,方便重命名###########
    id = int((img_dir[i].split('.')[0]).split('_')[1])
    img = Image.open(path_img + img_dir[i])
    size_img = img.size
    print(size_img)
    x = 0
    y = 0
    ########这里需要均匀裁剪几张，就除以根号下多少，这里我需要裁剪25张-》根号25=5（5*5）####
    w = int(size_img[0] / 4)
    h = int(size_img[0] / 4)
    for k in range(5):
        for v in range(5):
            region = img.crop((x + k * w, y + v * h, x + w * (k + 1), y + h * (v + 1)))
            #####保存图片的位置以及图片名称###############
            region.save('./new_1/' + 'gen1' + '%d%d' % (k, v) + '_%d' % id + '.jpg')
