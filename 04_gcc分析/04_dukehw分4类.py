import os

path = r"D:\06_scientific research\05_gee_labels\06_one_site\03_dukehw分4段\01_全部txt未序列 - 副本\\"
#
files = os.listdir(path)

for file in files:
    # print(file)
    month = file[12:14]
    if month == "03" or month == "04":
        text = path+file
        print(text)
        with open(text,"w+")as f:
            lines = f.readline()
            # print(lines)
            ss = lines[0:1]
            # print(ss)
            # for ss in s:
            new_line = lines.replace(ss, str("SOS"))
            print(new_line)

            list = str([new_line, "2 95 224 212"])
            # print(str(list))
            # result = flatten(list)
            result = list.replace("'", "")
            # print(result)
            result2 = result.replace(",", "")
            # print(result2)
            result3 = result2.replace("[", "")
            # print(result3)
            result4 = result3.replace("]", "")
            print(result4)
            f.write(result4)
    elif month == "09" or month == "10":
        text = path+file
        print(text)
        with open(text,"w+")as f:
            lines = f.readline()
            # print(lines)
            ss = lines[0:1]
            # print(ss)
            # for ss in s:
            new_line = lines.replace(ss, str("EOS"))
            print(new_line)

            list = str([new_line, "2 95 224 212"])
            # print(str(list))
            # result = flatten(list)
            result = list.replace("'", "")
            # print(result)
            result2 = result.replace(",", "")
            # print(result2)
            result3 = result2.replace("[", "")
            # print(result3)
            result4 = result3.replace("]", "")
            print(result4)
            f.write(result4)
    elif month == "07" or month == "08":
        text = path + file
        print(text)
        with open(text, "w+")as f:
            lines = f.readline()
            # print(lines)
            ss = lines[0:1]
            # print(ss)
            # for ss in s:
            new_line = lines.replace(ss, str("POP"))
            print(new_line)

            list = str([new_line, "2 95 224 212"])
            # print(str(list))
            # result = flatten(list)
            result = list.replace("'", "")
            # print(result)
            result2 = result.replace(",", "")
            # print(result2)
            result3 = result2.replace("[", "")
            # print(result3)
            result4 = result3.replace("]", "")
            print(result4)

            f.write(result4)