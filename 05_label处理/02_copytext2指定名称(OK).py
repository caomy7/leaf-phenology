
import shutil
#这个库复制文件比较省事

def objFileName():
    '''
    生成文件名列表
    :return:
    '''
    local_file_name_list = r'E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\01_image\LIST.TXT'
    #指定名单
    obj_name_list = []
    for i in open(local_file_name_list,'r'):
        obj_name_list.append(i.replace('\n',''))
    return obj_name_list

def copy_img():
    '''
    复制、重命名、粘贴文件
    :return:
    '''
    local_img_name=r'E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\04_labels\000001.txt'
    #指定要复制的图片
    path = r"E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\061_text"
    #指定存放图片的目录
    for i in objFileName():
        new_obj_name = i+'.txt'
        shutil.copy(local_img_name , path + '/' + new_obj_name)
        # shutil.copy(local_img_name,path+ ' / ' +new_obj_name)

if __name__ == '__main__':
    copy_img()