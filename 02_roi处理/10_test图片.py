import cv2

file = r"D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_github-code\00_roi\M\val_labels\100002.jpg"

img = cv2.imread(file)
print(img.shape)