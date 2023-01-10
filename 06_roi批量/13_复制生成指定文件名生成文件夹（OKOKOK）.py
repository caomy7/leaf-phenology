import os,sys
import shutil
#这个库复制文件比较省事
path1 = r"D:\06_scientific research\04_roi_labels\03_56site"
path2 = r"E:\IDM_phenocam\09_56siteRGB"

dirs = os.listdir(path1)

list =[]
for file in dirs:
    list.append(file)
print(list)

def copy_img():
    # local_img_name=r"D:\06_scientific research\05_gee_labels\06_one_site\02_dukehw\03_txt\dukehw_2016_01_01_120109.txt"
    #指定要复制的图片
    # print(objFileName)
    for i in list:
        print(i)
        new_obj_name = i

        print(path2 + "\\" + new_obj_name)
        f = path2 + "\\" + new_obj_name

        os.mkdir(f)


if __name__ == '__main__':
    copy_img()
    # objFileName()