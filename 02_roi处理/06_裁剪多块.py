from PIL import Image
import os


def image_cut(path_name, id, box_w, box_h):
    # path_name:要剪切的图像的文件夹(路径)
    # id:图像的命名序号（按我提供的重命名方式命名的序号，参考：图片批量重命名）
    # box_w：要剪切的图形的宽度
    # box_h：要剪切的图形的高度
    name = os.path.join(path_name, str(id) + ".jpg")  # 要剪切图像的名字
    # im = Image.open(name)
    # im = TIFFOpen(name,"r")
    im = Image.open(name)
    # im_size = im.size # 像素大小
    width = im.size[0]
    height = im.size[1]

    if width > box_w & (height > box_h):
        for i in range(int(width / box_w)):
            for j in range(int(height / box_h)):
                # name_cut = os.path.join(path_name, 'img_' + str(id) + '_' + str(i) + '_' + str(j) + ".jpg")
                name_cut = os.path.join(save_name, 'img_' + str(id) + '_' + str(i) + '_' + str(j) + ".jpg")
                cm = im.crop(box=(i * box_w, j * box_h, (i + 1) * box_w, (j + 1) * box_h))
                cm.save(name_cut)
                if i + 1 == int(width / box_w):
                    # name_cut = os.path.join(path_name, 'img_' + str(id) + '_' + str(i + 1) + '_' + str(j) + ".jpg")
                    name_cut = os.path.join(save_name, 'img_' + str(id) + '_' + str(i + 1) + '_' + str(j) + ".jpg")
                    cm = im.crop(box=((i + 1) * box_w, (j * box_h), width, (j + 1) * box_h))
                    cm.save(name_cut)
                if j + 1 == int(height / box_h):
                    # name_cut = os.path.join(path_name, 'img_' + str(id) + '_' + str(i) + '_' + str(j + 1) + ".jpg")
                    name_cut = os.path.join(save_name, 'img_' + str(id) + '_' + str(i) + '_' + str(j + 1) + ".jpg")
                    cm = im.crop(box=(i * box_w, (j + 1) * box_h, (i + 1) * box_w, height))
                    cm.save(name_cut)
                if i + 1 == int(width / box_w) & j + 1 == int(height / box_h):
                    # name_cut = os.path.join(path_name, 'img_' + str(id) + '_' + str(i + 1) + '_' + str(j + 1) + ".jpg")
                    name_cut = os.path.join(save_name, 'img_' + str(id) + '_' + str(i + 1) + '_' + str(j + 1) + ".jpg")
                    cm = im.crop(box=((i + 1) * box_w, (j + 1) * box_h, width, height))
                    cm.save(name_cut)
                    print(name_cut)


path_name = r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\val"  # 要剪切的图像的文件夹路径
save_name = r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\val_1"   # 存储裁剪得到的图像
for item in os.listdir(path_name):
    id = item.split('.')[0]
    image_cut(path_name, id, box_w=224, box_h=224)