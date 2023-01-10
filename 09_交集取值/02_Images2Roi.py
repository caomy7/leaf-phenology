import cv2
import os
import numpy as np
from PIL import Image
from PIL import Image
import numpy
path =r"D:\06_scientific research\11_Image2ROI\01_dukehw_extract_Roi\dukehw_roi.jpg"
roi = cv2.imread(path)

# print(roi)

path_img = r"D:\06_scientific research\11_Image2ROI\02_dukehw_img2roi_7639\05_Dukehw_elimate7637\\"
files = os.listdir(path_img)
for i in files:
    image = path_img +i
    print(image)
    img = cv2.imread(image)

    print(img.shape,roi.shape)
    # print(img.type,roi.type)

    im = cv2.multiply(img, roi)
    print(im)

    # im = im.convert('RGB')
    # im = Image.fromarray(numpy.uint8(deconvolved))
    im = im[:,:,[2,1,0]]
    im = Image.fromarray(numpy.uint8(im))
    # im.show()

    save_path = r"D:\06_scientific research\11_Image2ROI\01_dukehw_extract_Roi\02_Dukehw_roi3\\" + i
    im.save(save_path)

