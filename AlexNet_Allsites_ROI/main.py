import torch
from dataset import trainDataLoader,validDataLoader,testDataLoader
from model import Net
from train import train


epochs = 500
# lr = 0.1
lr = 0.01

net = Net()

if torch.cuda.is_available():
    print('CUDA is available!  Training on GPU ...\n')
    net.cuda()

#optimizer = torch.optim.SGD(net.parameters(),lr=0.01)   #利用SGD优化器优化神经网络，传入参数，学习率0.5
optimizer = torch.optim.Adam(net.parameters(),lr=lr)   #利用SGD优化器优化神经网络，传入参数，学习率0.5
loss_func = torch.nn.MSELoss()   #利用均方差进行回归


for epoch in range(epochs):
    train(epoch, net, trainDataLoader, optimizer, loss_func, validDataLoader,testDataLoader)

print('trainDataLoader')




