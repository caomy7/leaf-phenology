import cv2
import os
import numpy as np
from PIL import Image
from PIL import Image
import numpy
path =r"D:\06_scientific research\11_Image2ROI\06_Allsites_roi01(OK)\worcester_DB_1000_05.jpg"
roi = cv2.imread(path)

# print(roi)

path_img = r"D:\06_scientific research\11_Image2ROI\07_All45site_resize224_RGB\worcester\\"
files = os.listdir(path_img)
for i in files:
    image = path_img +i
    print(image)
    img = cv2.imread(image)

    print(img.shape,roi.shape)
    # print(img.type,roi.type)

    im = cv2.multiply(img, roi)
    print(im)

    im = im[:,:,[2,1,0]]
    im = Image.fromarray(numpy.uint8(im))
    # im.show()

    save_path = r"D:\06_scientific research\11_Image2ROI\08_AllsitsRGB_ROI\worcester\\" + i
    im.save(save_path)

