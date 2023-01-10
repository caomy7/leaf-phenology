import os
import shutil
path1 = r'D:\06_scientific research\11_Image2ROI\10_All45sites_ROIRGB14803\\'  #初始文件夹
path2 = r'D:\06_scientific research\11_Image2ROI\09_Allsites_RGB14453(OK)\\'  #比较的文件夹
save = r"D:\06_scientific research\11_Image2ROI\11_Allsites45_ROIRGB14453(OK)"

list1 = []
list2 = []
dict1 = os.listdir(path1)
# print(dict1)
for file1 in dict1:

    f1 = path1 +file1
    list1.append(file1)
    # print(f1)
dict2 = os.listdir(path2)
print(dict2)
for file2 in dict2:
    f2 = path2 + file2
    list2.append(file2)
    # print(f2)
    # print(file2)
for i in list1:
    if i in list2:
        print(i)
        old = path1 + i
        new = save +"\\"+ i
        print(old,new)
        shutil.copy(old,new)




