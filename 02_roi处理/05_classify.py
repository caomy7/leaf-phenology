import os, shutil
# 我们将重点讨论猫狗图像分类，数据集中包含 4000 张猫和狗的图像
# （2000 张猫的图像， 2000 张狗的图像）。我们将 2000 张图像用于训练， 1000 张用于验证， 1000
# 张用于测试。


# The path to the directory where the original
# dataset was uncompressed
# 原始数据集解压目录的路径
original_dataset_dir = r'D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_train_roi\01train\03_trainData\4个站点\label\label3'
# The directory where we will
# store our smaller dataset
# 保存较小数据集的目录
base_dir = r'D:\02_post_gratuation\06_deeplearning\05_demo\10_phenocam\01_train_roi\01train\03_trainData\4个站点\label\labelnew'
# os.mkdir(base_dir)

# 分别对应划分后的训练、# 验证和测试的目录
# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)


# 猫的训练图像目录
# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)


# 狗的训练图像目录
# Directory with our training dog pictures
# train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)

# 猫的验证图像目录
# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cats_dir)

# 狗的验证图像目录
# Directory with our validation dog pictures
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dogs_dir)

# 猫的测试图像目录
# Directory with our validation cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)

# 狗的测试图像目录
# Directory with our validation dog pictures
# test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)

# 将前 1000 张猫的图像复制
# 到 train_cats_dir
# Copy first 1000 cat images to train_cats_dir
fnames = ['1.{}.jpg'.format(i) for i in range(500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)


#     将接下来 500 张猫的图像复
# 制到 validation_cats_dir
# Copy next 500 cat images to validation_cats_dir
fnames = ['{}.jpg'.format(i) for i in range(500, 700)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# 将接下来的 500 张猫的图像
# 复制到 test_cats_dir
# Copy next 500 cat images to test_cats_dir
fnames = ['{}.jpg'.format(i) for i in range(700, 1063)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

#
# # Copy first 1000 dog images to train_dogs_dir
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
# # Copy next 500 dog images to validation_dogs_dir
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
# # Copy next 500 dog images to test_dogs_dir
# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)
# # As a
# # sanity
# # check, let
# # 's count how many pictures we have in each training split (train/validation/test):
#
# # 我们来检查一下，看看每个分组（训练 / 验证 / 测试）中分别包含多少张图像

print('total training cat images:', len(os.listdir(train_cats_dir)))
# total training cat images: 1000
# print('total training dog images:', len(os.listdir(train_dogs_dir)))
# total
# training
# dog
# images: 1000
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
# total
# validation
# cat
# images: 500
# print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
# total
# validation
# dog
# images: 500
print('total test cat images:', len(os.listdir(test_cats_dir)))
# total
# test
# cat
# images: 500
# print('total test dog images:', len(os.listdir(test_dogs_dir)))
# total
# test
# dog
# images: 500


