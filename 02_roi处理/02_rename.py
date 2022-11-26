# -*- coding:utf8 -*-

import os

class BatchRename():
    '''
    批量重命名文件夹中的图片文件

    '''
    def __init__(self):
        self.path = r'D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_train_roi\01train\03_trainData\4个站点\train\harvarbarn2'



    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 300000
        for item in filelist:
            # if item.endswith('.jpg'):
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), str(i) + '.jpg')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()