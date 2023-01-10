# -*- coding: utf-8 -*-
# 利用 np.hstack、np.vstack实现一幅图像中显示多幅图片

import cv2
from pylab import *
path= r"D:\06_scientific research\07_result\01-GCC分布图\02-roi_transform\04_roi_clip\\"

img3 = cv2.imread(path +'1_bullshoals_autumn.tif')
img1 = cv2.imread(path +'1_bullshoals_spring.tif')
img2 = cv2.imread(path +'1_bullshoals_summer.tif')
img4 = cv2.imread(path +'1_bullshoals_winter.tif')


img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
# htitch= np.hstack((img1, img2,img3,img4))
# # vtitch = np.vstack((img1, img3))
# cv2.imshow("test1",htitch)
# # cv2.imshow("test2",vtitch)
# plt.subplot(5,1)
# plt.imshow(img1,img2,img3,img4)
fig = plt.figure()
subplot(221)
imshow(img1)
title('img1')
axis('off')
subplot(222)
imshow(img2)
title('img2')
axis('off')
subplot(223)
imshow(img3)
title('img3')
axis('off')
subplot(224)
imshow(img4)
title('img4')
axis('off')

cv2.waitKey(0)
cv2.destroyAllWindows()
plt.show()