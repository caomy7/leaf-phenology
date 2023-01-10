import os,sys
import shutil
import time
#这个库复制文件比较省事

# path1 = r"C:\01_data\phenocam\01_roi_labels\07_resize2datasets\dataset\val"
# path1 = r"C:\01_data\phenocam\01_roi_labels\08_unresize2datasets\test_labels"
path1 = r"C:\01_data\phenocam\01_roi_labels\08_unresize2datasets\val_labels"
# path2 = r"C:\01_data\phenocam\01_roi_labels\07_resize2datasets\dataset\train_labels"

# path2 = r"C:\01_data\phenocam\01_roi_labels\08_unresize2datasets\datasets\test"
path2 = r"C:\01_data\phenocam\01_roi_labels\08_unresize2datasets\datasets\val"

# path2 = r"C:\01_data\phenocam\01_roi_labels\07_resize2datasets\dataset\train_labels"

def objFileName():
    dirs = os.listdir(path1)
    return dirs

def copy_img():
    # print(objFileName)
    for i in objFileName():
        # print(path1 + i)
        old_name,extension = os.path.splitext(i)

        # path3 = r"D:\02_Data\phenocamV2\04_roi_labels\05_unresize_all_dataset\train"
        path3 = r"C:\01_data\phenocam\01_roi_labels\05_unresize_all_dataset\05_unresize_all_dataset\train"
        local_img_name = os.listdir(path3)
        for j in local_img_name:
            if i == j:
                old = path3 + "\\" +old_name+".jpg"
                new = path2 + "\\" +old_name+".jpg"
                print(old)
                print(new)
                # shutil.copy(path3 + "\\" +old_name,path2 + "\\" +old_name)
                shutil.copy(old,new)
end = time.clock()


if __name__ == '__main__':
    start = time.clock()
    copy_img()
    print(" running time: %s second" % (end - start))

