import torch
import os
from dataset import trainDataLoader,validDataLoader,testDataLoader
# from model import Net
from model import ResNet
from model import resnet50
from model import resnet101
# from train import train,test
from train import train

from torchvision import *
# from keras.callbacks import ReduceLROnPlateau

from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DIVICES"] = "1"

# os.environ["CUDA_VISIBLE_DIVICES"] = "0"

# ids = 0,1

batch_size = 256
# batch_size = 32
# epochs = 100
epochs = 500
# lr = 0.1
lr = 0.0001

# net = ResNet()
# net = resnet50()
net = resnet101()
# resnet50 = models.resnet50(pretrained=True)

if torch.cuda.is_available():
    print('CUDA is available!  Training on GPU ...\n')
    net.cuda()

# optimizer = torch.optim.SGD(net.parameters(),lr=lr)   #利用SGD优化器优化神经网络，传入参数，学习率0.5
optimizer = torch.optim.Adam(net.parameters(),lr=lr)   #利用SGD优化器优化神经网络，传入参数，学习率0.5

# scheduler = lr_scheduler.ExponentialLR(optimizer,gamma = 0.9)
# print(lr)
# loss_func = torch.nn.MSELoss()   #利用均方差进行回归，分类用另一个函数
# plt.figure()
# x = list(range(100))
# y = []
# for epoch in range(100):
#     scheduler.step()
#     print(epoch,'lr ={:.6f}'.format(scheduler.get_lr()[0]))
#     y.append(scheduler.get_lr()[0])
# plt.plot(x,y)
# plt.show()

loss_func = torch.nn.CrossEntropyLoss()   #利用均方差进行回归，分类用另一个函数

for epoch in range(epochs):
    train(epoch, net, trainDataLoader, optimizer, loss_func, validDataLoader,testDataLoader)

print('trainDataLoader')


# test(net, testDataLoader,loss_func)


