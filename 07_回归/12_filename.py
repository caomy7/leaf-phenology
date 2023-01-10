import os

# path = r"E:\IDM_phenocam\08_dukehwRGB-IR\03_dukehw_IR8427\\"
# path = r"D:\06_scientific research\04_roi_labels\03_56site\tonzi\\"
path = r"D:\06_scientific research\04_roi_labels\03_56site\acadia\\"
# path2 = r"C:\01_data\08_dukehw8298\\"
path2 = r"D:\06_scientific research\10_回归\02_56siteRGB\acadia\\"
# path2 = r"D:\06_scientific research\10_回归\02_56siteRGB\tonzi\\"
files = os.listdir(path)
for file in files:
    print(file)

    # month = str(file[14:16])
    month = str(file[12:14])
    # month = str(file[17:19])
    # month = str(file[22:24])
    # data = str(file[20:22])
    data = str(file[15:17])
    # data = str(file[25:27])
    # no = str(file[28:])
    no = str(file[18:])
    # no = str(file[27:])

    print(month)
    print(data)
    print(no)
    old_name = path+file
    l = []
    if month == "05" and data >= "08":
        time = int(data) - 7
            # print(data)
        new_name = path2 + str(time) + "_" + file
        print(old_name, new_name)
        print(time)
        # l= l.append(time)
        # print(sum(l))
        os.rename(old_name, new_name)
    # elif month == "03" :
    #     time = int(data) +1
    #     print(time)
    #     new_name = path2 + str(time) + "_" + file
    #     print(old_name, new_name)
    #     # os.chdir(path2)
    #     os.rename(old_name, new_name)
    #
    # elif month == "04" :
    #     time = int(data) + 7
    #     print(time)
    #     new_name = path2 + str(time) + "_" + file
    #     print(old_name, new_name)
    #     # os.chdir(path2)
    #     os.rename(old_name, new_name)
    #
    # elif month == "05":
    #     time = int(data) + 10
    #     new_name = path2 + str(time) + "_" + file
    #     print(old_name, new_name)
    #     # os.chdir(path2)
    #     os.rename(old_name, new_name)

    elif month == "06":
        time = int(data) + 24
        new_name = path2 + str(time) + "_" + file
        print(old_name, new_name)
        # os.chdir(path2)
        os.rename(old_name, new_name)

    elif month == "07":
        time = int(data) + 54
        new_name = path2 + str(time) + "_" + file
        print(old_name, new_name)
        # os.chdir(path2)
        os.rename(old_name, new_name)

    elif month == "08":
        time = int(data) +85
        new_name = path2 + str(time) + "_" + file
        print(old_name, new_name)
        # os.chdir(path2)
        os.rename(old_name, new_name)

    elif month == "09":
        time = int(data) + 116
        new_name = path2 + str(time) + "_" + file
        print(old_name, new_name)
        # os.chdir(path2)
        os.rename(old_name, new_name)

    # elif month == "10" :
    #     time = int(data) + 146
    #     print(month,time)
    #     new_name = path2 + str(time) + "_" + file
    #     print(old_name, new_name)
    #     os.chdir(path2)
    #     os.rename(old_name, new_name)
    # elif month == "10":
    #     time = int(data) + 152
    #     print( month, time)
    #     new_name = path2 + str(time) + "_" + file
    #     print(old_name, new_name)
    #     # os.chdir(path2)
    #     os.rename(old_name, new_name)

    elif month == "10" and data <= "21":
        time = int(data) + 146
        print(month,time)
        new_name = path2 + str(time) + "_" + file
        print(old_name, new_name)
        # os.chdir(path2)
        os.rename(old_name, new_name)

    else:
        time = str("0")
        new_name = path2 +str(time) +"_"+file
        print(old_name,new_name)
        # os.chdir(path2)
        os.rename(old_name,new_name)