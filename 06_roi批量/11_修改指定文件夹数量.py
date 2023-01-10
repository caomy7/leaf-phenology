import os

path = r"D:\06_scientific research\05_gee_labels\06_one_site\02_dukehw\03_txt\10 - 副本\\"
files = os.listdir(path)

for file in files:
    print(file)
    name_num = int(file[15:17])
    print(name_num)

    # line =[]

    with open(path +file,"w+") as txt:
        f =txt.readline()
        # print(f)
        ff = f[0:1]
        # if name_num >12:
        i = int(name_num)+199
        line = f.replace(ff, str(i))
    # print(line)
        m = [line,"2 95 224 212"]
        print(m)

        result = str(m).replace("'", "")
        result1 = str(result).replace(",", "")
        result2 = str(result1).replace("[", "")
        result3 = str(result2).replace("]", "")
        print(result3)

        txt.write(result3)