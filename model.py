import torch
import torch.nn as nn

def conv3x3(in_channels,out_channels,stride=1):
    return torch.nn.Sequential(torch.nn.ReplicationPad2d(1),torch.nn.Conv2d(in_channels,out_channels,3,stride=stride,padding=1))
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Net(torch.nn.Module):
    def __init__(self):   #搭建这些层需要的信息
        super(Net,self).__init__()   #进行继承

        self.conv1 = conv3x3(3,  64)
        self.conv2 = conv3x3(64, 128)
        self.conv3 = conv3x3(128, 256)
        self.conv4 = conv3x3(256, 512)

        self.fc1 = torch.nn.Linear(512*5*5,128)
        # self.fc1 = torch.nn.Linear(512 * 4 * 4 , 128)
        # self.fc2 = torch.nn.Linear(128 , 56)
        self.fc2 = torch.nn.Linear(128 , 64)
        self.fc3 = torch.nn.Linear(64 , 1)
        # self.fc3 = torch.nn.Linear(56 , 236)


    def forward(self,x):    #前向传播，每个图链接起来。。注意不要忘记了x,一共两个参数
        out = torch.nn.functional.relu(self.conv1(x))  #  64 * 64 *64
        out = torch.nn.functional.max_pool2d(out , 2)  #  32 * 32 *64
        out = torch.nn.functional.relu(self.conv2(out))  #  32 * 32 *128
        out = torch.nn.functional.max_pool2d(out , 2)  # 16 * 16 *128
        out = torch.nn.functional.relu(self.conv3(out))
        out = torch.nn.functional.max_pool2d(out , 2)  # 8 * 8*256
        out = torch.nn.functional.relu(self.conv4(out))
        out = torch.nn.functional.max_pool2d(out, 2)  #4 * 4  *512

        out = out.view(out.size(0) , -1)  # flatten
        out = torch.nn.functional.relu(self.fc1(out))  #
        out = torch.nn.functional.relu(self.fc2(out))  #
        out = self.fc3(out)  # 5 classes
        return out
