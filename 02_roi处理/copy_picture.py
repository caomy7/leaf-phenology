"""
前提 当前目录下有客户想要的文件
需求：
1.输入文件名
2.用户输入要产生多少份文件，则生成多少份
3.生成文件名为ReadX.wav,其中X为数字
"""
import os
# file_path = os.path.dirname(r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_train_roi\01train\03_trainData\4个站点\label\harvardbarn2")
os.system("start explorer D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_train_roi\01train\03_trainData\4个站点\label\harvardbarn2")

print(os.path.abspath(os.path.dirname(__file__)))

old_name = input("请输入模板文件名文件名：")
num = int(input("请输入您要复制的文件份数："))


index1 = old_name.rfind('.')  # 识别文件 .的位置
print(index1)
first1_name = old_name[:index1]  # 取文件名.前面的字符串

last_name = old_name[index1:]  # 取文件名.后面的字符串

i = 0
while True:
    if i < num:
        # 创建文件名为文件名前面+数字+文件名后缀，例如输入文件名为Read.wav ，产生1份,最终文件名为：Read1.wav
        new_name = '{my_first_name}{my_i}{my_last_name}'.format(my_first_name=first1_name, my_i=i,
                                                                my_last_name=last_name)
        new_f = open(new_name, 'w')  # 创建文件
        # 打开旧文件
        old_f = open(old_name, 'rb')
        # 打开新文件
        new_f = open(new_name, 'wb')
        # 拷贝旧文件内容到新文件，每次拷贝1024字节，直到拷贝结束
        while True:
            con = old_f.read(1024)
            if len(con) == 0:
                break
            new_f.write(con)
        i += 1
    else:
        print(f"您产生的{num}份文件已经结束")
        break

new_f.close()  # 关闭新文件
old_f.close()  # 关闭旧文件