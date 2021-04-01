import torch
from dataset import trainDataLoader,validDataLoader,testDataLoader
# from model import Net
from model import ResNet
from model import resnet50
from train import train
import os


from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

# torch.cuda.empty_cache()
# os.environ["CUDA_VISIBLE_DIVICES"] = "1"
# ids = [0,1]

epochs =500
# lr = 0.1
lr = 0.1

# net = ResNet()
net = resnet50()
# net = torch.nn.DataParalle(net,device_ids = ids)


if torch.cuda.is_available():
    print('CUDA is available!  Training on GPU ...\n')
    net.cuda()

# optimizer = torch.optim.SGD(net.parameters(),lr=0.01)   #利用SGD优化器优化神经网络，传入参数，学习率0.5
optimizer = torch.optim.Adam(net.parameters(),lr=lr)   #利用SGD优化器优化神经网络，传入参数，学习率0.5
loss_func = torch.nn.MSELoss()   #利用均方差进行回归，分类用另一个函数
# loss_func = torch.nn.SmoothL1Loss()   #利用均方差进行回归，分类用另一个函数

# scheduler = lr_scheduler.ExponentialLR(optimizer,gamma = 0.9)
# scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[120,130,170,190,210,280,320,340],gamma= 0.1)
# plt.figure()
# x = list(range(100))
# y = []
# for epoch in range(100):
#     scheduler.step()
#     print(epoch,'lr ={:.6f}'.format(scheduler.get_lr()[0]))
#     y.append(scheduler.get_lr()[0])
# plt.plot(x,y)
# plt.show()

for epoch in range(epochs):
    train(epoch, net, trainDataLoader, optimizer, loss_func, validDataLoader,testDataLoader)

print('trainDataLoader')





