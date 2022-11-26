import cv2 as cv
import os

"""
    转换tiff格式为png + 横向裁剪tiff遥感影像图
"""
def Convert_To_Png_AndCut(dir):
    files = os.listdir(dir)
    ResultPath1 = r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\label3_png"  # 定义转换格式后的保存路径
    ResultPath2 = r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\label3_resize\\"  # 定义裁剪后的保存路径
    ResultPath3 = r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\label3_resize\\"  # 定义裁剪后的保存路径
    #
    # ResultPath1 = "./RS_ToPngDir/"  # 定义转换格式后的保存路径
    # ResultPath2 = "./RS_Cut_Result/"  # 定义裁剪后的保存路径
    # ResultPath3 = "./RS_Cut_Result/"  # 定义裁剪后的保存路径
    for file in files:  # 这里可以去掉for循环
        a, b = os.path.splitext(file)  # 拆分影像图的文件名称
        # print(a,b)
        this_dir = os.path.join(dir +"\\"+ file)  # 构建保存 路径+文件名
        # print(this_dir)

        img = cv.imread(this_dir, 1)  # 读取tif影像
        # print(img)
        # 第二个参数是通道数和位深的参数，
        # IMREAD_UNCHANGED = -1  # 不进行转化，比如保存为了16位的图片，读取出来仍然为16位。
        # IMREAD_GRAYSCALE = 0  # 进行转化为灰度图，比如保存为了16位的图片，读取出来为8位，类型为CV_8UC1。
        # IMREAD_COLOR = 1   # 进行转化为RGB三通道图像，图像深度转为8位
        # IMREAD_ANYDEPTH = 2  # 保持图像深度不变，进行转化为灰度图。
        # IMREAD_ANYCOLOR = 4  # 若图像通道数小于等于3，则保持原通道数不变；若通道数大于3则只取取前三个通道。图像深度转为8位

        cv.imwrite(ResultPath1 + a + "_" + ".png", img)  # 保存为png格式
        print("ok")

        # 下面开始裁剪-不需要裁剪tiff格式的可以直接注释掉
        hight = img.shape[0]  # opencv写法，获取宽和高
        width = img.shape[1]

        new_im = cv.resize(img, (224, 224))

        cv.imwrite(ResultPath2 + a + "_" + b, new_im)
        print("finish")


"""
    横向裁剪PNG图
"""
#
#
# def toCutPng(dir):
#     files = os.listdir(dir)
#     ResultPath = "./RS_CutPng_Result/"  # 定义裁剪后的保存路径
#     for file in files:
#         a, b = os.path.splitext(file)  # 拆分影像图的文件名称
#         this_dir = os.path.join(dir + file)
#         img = Image.open(this_dir)  # 按顺序打开某图片
#         width, hight = img.size
#         w = 480  # 宽度
#         h = 360  # 高度
#         _id = 1  # 裁剪结果保存文件名：0 - N 升序方式
#         y = 0
#         while (y + h <= hight):  # 控制高度,图像多余固定尺寸总和部分不要了
#             x = 0
#             while (x + w <= width):  # 控制宽度，图像多余固定尺寸总和部分不要了
#                 new_img = img.crop((x, y, x + w, y + h))
#                 new_img.save(ResultPath + a + "_" + str(_id) + b)
#                 _id += 1
#                 x += w
#             y = y + h
#

if __name__ == '__main__':
    # _path = r"./RS_TiffDir/"  # 遥感tiff影像所在路径
    _path = r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\label3"  # 遥感tiff影像所在路径

    # 裁剪影像图
    Convert_To_Png_AndCut(_path)
