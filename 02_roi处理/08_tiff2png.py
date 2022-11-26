from wand.image import Image
import os
# from PIL import Image # 一开始在这里报错是因为import 一个文件的时候，不能重名，在windows下需要安装一个exe
def get_imlist(path):
    """返回目录中所有tif图像的文件名列表"""
    # return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(".tif")]
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(".jpg")]
if __name__ == '__main__':
    path = "D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\test_labels"
    listdir = get_imlist(path)

    for dir in listdir:
        print(dir)
        with Image(filename = str(dir)) as img:
            img.resize(224,224) # width, height
            # 存的目录为"G:/Test/6-28/HBsAg_png/",用了一步replace，换了个目录
            # img.save(filename = (str(dir)[:-3]+'png').replace("HBsAg_tif","HBsAg_png")) # png, jpg, bmp, gif, tiff All OK--
            img.save(filename = (str(dir)[:-3]+'jpg').replace("HBsAg_tif","HBsAg_jpg")) # png, jpg, bmp, gif, tiff All OK--
