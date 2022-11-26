import cv2
import numpy as np 
import os
# from osgeo import gdal

# root_dir = r'E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\03_roitif\\'
# txt_dir = r'E:\phenicam\02_gcc\01_acadia\acadia_DB_1000\03_roitif\txt\\'
root_dir = r'E:\phenicam\02_gcc\04_45site_resize\02_roi\\'
txt_dir = r'E:\phenicam\02_gcc\05_labels_noresize\\'
# images = ['acadia_DB_2000_02.tif','ahwahnee_GR_1000_01.tif']
# images = ['26.tif','27.tif']
# print(images)
images = os.listdir(root_dir)

print(images)
#read image
for image in images:
    # img = cv2.imread(root_dir+image)
    print(root_dir)
    print(image)
    img = cv2.imread(root_dir+image)
    # img = cv2.imshow("image",image)
    print(img)
    # find the contours first （Input img should be 1 channel）    或
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # print(gray)
    # ret , binary = cv2.threshold(gray , 0 , 255 , cv2.THRESH_BINARY)
    contours,hier = cv2.findContours(img[:,:,0],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   #查找轮廓
    # contours,hier = cv2.findContours(binary[:,:,0],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   #查找轮廓
    #image,contours,hier=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  #for opencv3

    print(hier , contours)
    height,width = img[:,:,0].shape

    boxes = []
    if contours:
        txt = open(txt_dir+image[:-4]+'.txt','w')
        for contour in contours:
          x,y,w,h = cv2.boundingRect(contour) # get top-left(X,Y) and height and width
          
          # top-left: tl  bottom-right:tr
          expand = 0
          xmin = max(0,x-expand)
          ymin = max(0,y-expand)  
          xmax = min(width,x+w+expand)
          ymax = min(height,y+h+expand)
          
          boxes.append([xmin,ymin,xmax,ymax])
          # txt.write('{} {} {} {} {}\n'.format("0",xmin,ymin,xmax,ymax))


        #print(boxes)
        txt.close()