import os,sys
import shutil
#这个库复制文件比较省事
path1 = r"D:\06_scientific research\05_gee_labels\06_one_site\02_dukehw\01_2016\01"
path2 = r"D:\06_scientific research\05_gee_labels\06_one_site\02_dukehw\03_txt\01"

def objFileName():
    dirs = os.listdir(path1)
    return dirs

list =[]

def copy_img():
    local_img_name=r"D:\06_scientific research\05_gee_labels\06_one_site\02_dukehw\03_txt\dukehw_2016_01_01_120109.txt"
    #指定要复制的图片
    # print(objFileName)
    for i in objFileName():
        # print(i)
        old_name,extension = os.path.splitext(i)
        # print(old_name)

        new_obj_name = old_name+".txt"
        # new_obj_name = i
        # print(new_obj_name)
        # print(local_img_name)
        print(path2 + "\\" + new_obj_name)
        if local_img_name == i:
            shutil.copy(local_img_name, path2 + "\\"+new_obj_name)
    # os.remove(local_img_name)

if __name__ == '__main__':
    copy_img()
    # objFileName()