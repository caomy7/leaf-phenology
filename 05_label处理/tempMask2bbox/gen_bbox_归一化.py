import cv2
import numpy as np 
import os
# from osgeo import gdal

root_dir = r'E:\phenicam\02_gcc\04_45site_resize\02_roi\acadia\\'
txt_dir = r'E:\phenicam\02_gcc\05_labels_noresize\\'

images = os.listdir(root_dir)
# print(images)

for image in images:
    img = cv2.imread(root_dir+image)
    # img = cv2.imread(image)
    print(img)
    # cv2.imshow("img",img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret , binary = cv2.threshold(gray , 0 , 255 , cv2.THRESH_BINARY)
    # find the contours first （Input img should be 1 channel）    或  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # contours,hier = cv2.findContours(img[:,:,1],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   #查找轮廓
    # contours,hier = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   #查找轮廓
    contours,hier = cv2.findContours(binary.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   #查找轮廓
    #image,contours,hier=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  #for opencv3
    print(hier,contours)
    for c in contours:
        cv2.drawContours(img,[c],-1,(255,0,0),2)
        cv2.imshow("c",img)
        cv2.waitKey(123)
    # cv2.imshow("contours",contours)
    # cv2.imread("contours",contours)

    height,width = img[:,:,0].shape  

    boxes = []
    if contours:
        txt = open(txt_dir+image[:-4]+'.txt','w')
        for contour in contours:
          x,y,w,h = cv2.boundingRect(contour) # get top-left(X,Y) and height and width
          # print(x,y,w,h)
          
          # top-left: tl  bottom-right:tr
          expand = 0
          xmin = max(0,x-expand)
          ymin = max(0,y-expand)  
          xmax = min(width,x+w+expand)
          ymax = min(height,y+h+expand)

          # dw = 1 / width
          # dh = 1 / height
          # x_new = (xmin + xmax) / 2 * dw
          # y_new = (ymin + ymax) / 2 * dh
          # w_new = (xmax - xmin) / 2 * dw
          # h_new = (ymax - ymin) / 2 * dh




          boxes.append([xmin,ymin,xmax,ymax])
          # boxes.append([x_new,y_new,w_new,h_new])
          # txt.write('{} {} {} {} {}\n'.format(xmin,ymin,xmax,ymax,'0'))
          # txt.write('{} {} {} {} {}\n'.format("0",x_new,y_new,w_new,h_new))


        #print(boxes)
        txt.close()