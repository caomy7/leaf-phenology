import torch
import torch.nn as nn

def conv3x3(in_channels,out_channels,stride=1):
    return torch.nn.Sequential(torch.nn.ReplicationPad2d(1),torch.nn.Conv2d(in_channels,out_channels,3,stride=stride))
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 裁剪roi区域训练，dukehw[2:224, 95:212]

class Net(torch.nn.Module):
    def __init__(self):   #搭建这些层需要的信息
        super(Net,self).__init__()   #进行继承,输入为224*224*3
        # print(Net)

        self.conv1 = nn.Conv2d(3,64,3,1,1)    #224*224*64
        self.conv2 = nn.Conv2d(64, 64,3,1,1)    #224*224*64

        self.conv3 = nn.Conv2d(64, 128,3,1,1)   #112*112*128
        self.conv4 = nn.Conv2d(128, 128,3,1,1)   #112*112*128

        self.conv5 = nn.Conv2d(128,256,3,1,1)    #56*56*256
        self.conv6 = nn.Conv2d(256, 256,3,1,1)
        self.conv7 = nn.Conv2d(256, 256,3,1,1)

        self.conv8 = nn.Conv2d(256, 512,3,1,1)   #28*28*512
        self.conv9 = nn.Conv2d(512, 512,3,1,1)
        self.conv10 = nn.Conv2d(512, 512,3,1,1)

        self.conv11 = nn.Conv2d(512, 512,3,1,1)   #14*14*512
        self.conv12 = nn.Conv2d(512, 512,3,1,1)
        self.conv13 = nn.Conv2d(512, 512,3,1,1)
        # self.conv14 = conv3x3(512, 512)

        # self.fc1 = torch.nn.Linear(512 * 7 * 4 , 128)
        # self.fc1 = torch.nn.Linear(512 * 2 * 2 , 128)
        self.fc1 = torch.nn.Linear(512 * 2 * 2 , 128)
        nn.BatchNorm2d(1)

        nn.ReLU(True)
        nn.Dropout()

        self.fc2 = torch.nn.Linear(128 , 64)    #4096
        nn.BatchNorm2d(1)

        nn.ReLU(True)
        nn.Dropout()
        # self.fc3 = torch.nn.Linear(64 , 1)
        # self.fc3 = torch.nn.Linear(64 , 46)
        # self.fc3 = torch.nn.Linear(64 , 236)
        # self.fc3 = torch.nn.Linear(64 , 314)
        self.fc3 = torch.nn.Linear(64 , 1)

    def forward(self, x):  # 前向传播，每个图链接起来。。注意不要忘记了x,一共两个参数
        out = torch.nn.functional.relu(self.conv1(x))  # 64 * 117*64        # print(out.shape)
        out = torch.nn.functional.relu(self.conv2(out))  # 222 * 117*64
        out = torch.nn.functional.max_pool2d(out, 2)  # 32 * 59*64
        out = torch.nn.functional.relu(self.conv3(out))  # 111 * 59*128
        out = torch.nn.functional.relu(self.conv4(out))  # 111 * 59*128
        out = torch.nn.functional.max_pool2d(out, 2)  # 16* 30*128
        out = torch.nn.functional.relu(self.conv5(out))  # 56 * 30*256
        out = torch.nn.functional.relu(self.conv6(out))  # 56 * 30*256
        out = torch.nn.functional.relu(self.conv7(out))  # 56 * 30*256
        out = torch.nn.functional.max_pool2d(out, 2)  # 8 * 15*256
        out = torch.nn.functional.relu(self.conv8(out))  # 28*15*512
        out = torch.nn.functional.relu(self.conv9(out))  # 28*15*512
        out = torch.nn.functional.relu(self.conv10(out))  # 28*15*512
        out = torch.nn.functional.max_pool2d(out, 2)  # 4 * 8*512
        out = torch.nn.functional.relu(self.conv11(out))  # 14 * 8*512
        out = torch.nn.functional.relu(self.conv12(out))  # 14 * 8*512
        out = torch.nn.functional.relu(self.conv13(out))  # 14 * 8*512


        out = torch.nn.functional.max_pool2d(out, 2)  # 512* 7 * 4   2

        out = out.view(out.size(0), -1)  # flatten
        out = torch.nn.functional.relu(self.fc1(out))  #
        # out = torch.nn.Dropout(torch.nn.functional.relu(self.fc1(out)))
        # out = torch.nn.Dropout(torch.nn.functional.relu(self.fc2(out)))
        out = torch.nn.functional.relu(self.fc2(out))  #

        out = self.fc3(out)  #
        return out


#未裁剪，整个224*224进行训练
# class Net(torch.nn.Module):
#     def __init__(self):   #搭建这些层需要的信息
#         super(Net,self).__init__()   #进行继承,输入为224*224*3
#         # print(Net)
#
#         self.conv1 = nn.Conv2d(3,64,3,1,1)    #224*224*64
#         self.conv2 = nn.Conv2d(64, 64,3,1,1)    #224*224*64
#
#         self.conv3 = nn.Conv2d(64, 128,3,1,1)   #112*112*128
#         self.conv4 = nn.Conv2d(128, 128,3,1,1)   #112*112*128
#
#         self.conv5 = nn.Conv2d(128,256,3,1,1)    #56*56*256
#         self.conv6 = nn.Conv2d(256, 256,3,1,1)
#         self.conv7 = nn.Conv2d(256, 256,3,1,1)
#
#         self.conv8 = nn.Conv2d(256, 512,3,1,1)   #28*28*512
#         self.conv9 = nn.Conv2d(512, 512,3,1,1)
#         self.conv10 = nn.Conv2d(512, 512,3,1,1)
#
#         self.conv11 = nn.Conv2d(512, 512,3,1,1)   #14*14*512
#         self.conv12 = nn.Conv2d(512, 512,3,1,1)
#         self.conv13 = nn.Conv2d(512, 512,3,1,1)
#         # self.conv14 = conv3x3(512, 512)
#
#
#         self.fc1 = torch.nn.Linear(512 * 7 * 7 , 128)
#         nn.ReLU(True)
#         nn.Dropout()
#         self.fc2 = torch.nn.Linear(128 , 64)    #4096
#         nn.ReLU(True)
#         nn.Dropout()
#         self.fc3 = torch.nn.Linear(64 , 1)
#
#
#     def forward(self,x):    #前向传播，每个图链接起来。。注意不要忘记了x,一共两个参数
#         out = torch.nn.functional.relu(self.conv1(x))  #  224 * 224*64
#         # print(out.shape)
#         out = torch.nn.functional.relu(self.conv2(out))   #  224 * 224*64
#         out = torch.nn.functional.max_pool2d(out , 2)   #  112 * 112*64
#         out = torch.nn.functional.relu(self.conv3(out)) #  112 * 112*128
#         out = torch.nn.functional.relu(self.conv4(out))  #  112 * 112*128
#         out = torch.nn.functional.max_pool2d(out , 2) #  56 * 56*128
#         out = torch.nn.functional.relu(self.conv5(out))   #  56 * 56*256
#         out = torch.nn.functional.relu(self.conv6(out))   #  56 * 56*256
#         out = torch.nn.functional.relu(self.conv7(out))    #  56 * 56*256
#         out = torch.nn.functional.max_pool2d(out , 2)  # 28 * 28*256
#         out = torch.nn.functional.relu(self.conv8(out))    #28*28*512
#         out = torch.nn.functional.relu(self.conv9(out))     #28*28*512
#         out = torch.nn.functional.relu(self.conv10(out))     #28*28*512
#         out = torch.nn.functional.max_pool2d(out, 2)  # 14 * 14*512
#         out = torch.nn.functional.relu(self.conv11(out)) # 14 * 14*512
#         out = torch.nn.functional.relu(self.conv12(out)) # 14 * 14*512
#         out = torch.nn.functional.relu(self.conv13(out)) # 14 * 14*512
#         out = torch.nn.functional.max_pool2d(out, 2)  # 512* 7 * 7
#
#
#         out = out.view(out.size(0) , -1)  # flatten
#         out = torch.nn.functional.relu(self.fc1(out))  #
#         # out = torch.nn.Dropout(torch.nn.functional.relu(self.fc1(out)))
#         # out = torch.nn.Dropout(torch.nn.functional.relu(self.fc2(out)))
#         out = torch.nn.functional.relu(self.fc2(out))  #
#         out = self.fc3(out)  # 5 classes
#         return out

#
# class Net(torch.nn.Module):
#     def __init__(self):   #搭建这些层需要的信息
#         super(Net,self).__init__()   #进行继承
#
#         self.conv1 = conv3x3(3,  64)
#         self.conv2 = conv3x3(64, 128)
#         self.conv3 = conv3x3(128, 256)
#         self.conv4 = conv3x3(256, 512)
#
#         self.fc1 = torch.nn.Linear(512 * 4 * 4 , 128)
#         self.fc2 = torch.nn.Linear(128 , 56)
#         # self.fc3   = torch.nn.Linear(40, 228)
#         self.fc3 = torch.nn.Linear(56 , 1)
#
#
#     def forward(self,x):    #前向传播，每个图链接起来。。注意不要忘记了x,一共两个参数
#         out = torch.nn.functional.relu(self.conv1(x))  #  64 * 64
#         out = torch.nn.functional.max_pool2d(out , 2)  #  32 * 32
#         out = torch.nn.functional.relu(self.conv2(out))  #  32 * 32
#         out = torch.nn.functional.max_pool2d(out , 2)  # 16 * 16
#         out = torch.nn.functional.relu(self.conv3(out))
#         out = torch.nn.functional.max_pool2d(out , 2)  # 8 * 8
#         out = torch.nn.functional.relu(self.conv4(out))
#         out = torch.nn.functional.max_pool2d(out, 2)  #512* 4 * 4
#
#         out = out.view(out.size(0) , -1)  # flatten
#         out = torch.nn.functional.relu(self.fc1(out))  #
#         out = torch.nn.functional.relu(self.fc2(out))  #
#         out = self.fc3(out)  # 5 classes
#         return out

#
# # Load CNN model
# def loadCNN(bTraining=False):
#     global get_output
#     model = nn.Sequential()
#
#     model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
#                      padding='valid',
#                      input_shape=(img_channels, img_rows, img_cols)))
#     convout1 = Activation('relu')
#     model.add(convout1)
#     model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
#     convout2 = Activation('relu')
#     model.add(convout2)
#     model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
#     model.add(Dropout(0.5))
#
#     model.add(Flatten())
#     model.add(Dense(128))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(nb_classes))
#     model.add(Activation('softmax'))
#
#     # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#
#     # Model summary
#     model.summary()
#     # Model conig details
#     model.get_config()
#
#     if not bTraining:
#         # List all the weight files available in current directory
#         WeightFileName = modlistdir('.', '.hdf5')
#         if len(WeightFileName) == 0:
#             print(
#                 'Error: No pretrained weight file found. Please either train the model or download one from the https://github.com/asingh33/CNNGestureRecognizer')
#             return 0
#         else:
#             print('Found these weight files - {}'.format(WeightFileName))
#         # Load pretrained weights
#         w = int(input("Which weight file to load (enter the INDEX of it, which starts from 0): "))
#         fname = WeightFileName[int(w)]
#         print("loading ", fname)
#         model.load_weights(fname)
#
#     # refer the last layer here
#     layer = model.layers[-1]
#     get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])
#
#     return model
#
# class CNN(torch.nn.Module):
#     def __init__(self):
#         super(CNN,self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(            #图片维度为 (1,28,28)
#                 in_channels=1,    #前面的层有多高，即几层。本次中为1，1个颜色层
#                 out_channels=16,     #有16个区域的特征，输出为高度16层的图片
#                 kernel_size=5,   #指filter过滤器为5*5的像素点进行扫描
#                 stride=1,       #指每隔多少步跳度
#                 padding=2,      #指图片扫描后，不够为一个图片则除掉。旁边围了一圈为0的数据。。   if stride = 1,padding = (kernel_size -1)/2 = (5-1)/2
#                                  #（16，28，28）
#             ),     #卷积层的信息过滤器，过滤器收集图片中的信息。包含宽度、长度和高度
#             nn.ReLU(),        #（16，28，28）
#             nn.MaxPool2d(kernel_size=2),   #筛选出重要的数据，选择2*2的数据图片 #（16，14，14）
#         )
#         self.conv2 = nn.Sequential(   #建立第二层
#             nn.Conv2d(16,32,5,1,2),    #16为输入接收层，加工为32层。5为kernel_size,1为stride,2为padding
#             nn.ReLU(),        #（32，14，14）
#             nn.MaxPool2d(2),    #输出为kernel_size=2不变    #（32，7，7）
#         )
#         self.out = nn.Linear(32*7 *7,10)
#
#     #展平数据
#     def forward(self,x):
#         x=self.conv1(x)
#         x=self.conv2(x)          #batch(32,7,7)
#         x=x.view(x.size(0),-1)       #batch(32*7*7),变成一维数据
#         output = self.out(x)     #输出
#         return output,x

# class Net(nn.Module):
#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None):
#         super(Net, self).__init__()
#         # 参数比调用多几个，模型相较于最初发文章的时候有过更新
#         # block: basicblock或者bottleneck，后续会提到
#         # layers：每个block的个数，如resnet50， layers=[3,4,6,3]
#         # num_classes: 数据库类别数量
#         # zero_init_residual：其他论文中提到的一点小trick，残差参数为0
#         # groups：卷积层分组，应该是为了resnext扩展
#         # width_per_group：同上，此外还可以是wideresnet扩展
#         # replace_stride_with_dilation：空洞卷积，非原论文内容
#         # norm_layer：原论文用BN，此处设为可自定义
#
#         # 中间部分代码省略，只看模型搭建部分
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
#                                        dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#     # 中间部分代码省略
#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             # 当需要特征图需要降维或通道数不匹配的时候调用
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         # 每一个self.layer的第一层需要调用downsample，所以单独写，跟下面range中的1 相对应
#         # block的定义看下文
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))
#
#         return nn.Sequential(*layers)
#
#     def _forward_impl(self, x):
#         # 前向传播
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x
#
# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)
#
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
# #用在resnet18中的结构，也就是两个3x3卷积
# class BasicBlock(nn.Module):
#     expansion = 1
#     __constants__ = ['downsample']
#     #inplanes：输入通道数
#     #planes：输出通道数
#     #base_width，dilation，norm_layer不在本文讨论范围
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(BasicBlock, self).__init__()
#         #中间部分省略
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         #为后续相加保存输入
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             #遇到降尺寸或者升维的时候要保证能够相加
#             identity = self.downsample(x)
#
#         out += identity#论文中最核心的部分，resnet的简洁和优美的体现
#         out = self.relu(out)
#
#         return out
#
# #bottleneck是应用在resnet50及其以上的结构，主要是1x1,3x3,1x1
# class Bottleneck(nn.Module):
#     expansion = 4
#     __constants__ = ['downsample']
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(Bottleneck, self).__init__()
#         #中间省略
#         self.conv1 = conv1x1(inplanes, planes)
#         self.bn1 = norm_layer(planes)
#         self.conv2 = conv3x3(planes, planes, stride, groups, dilation)
#         self.bn2 = norm_layer(planes)
#         self.conv3 = conv1x1(planes, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#     #同basicblock
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out


# torch.nn.Module.apply(fn)
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#VGGNet
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()  # 224, 224, 3
#
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 1, padding=1),  # 224, 224, 64
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, 3, 1, padding=1),  # 224, 224, 64
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # 112, 112, 64
#             nn.Conv2d(64, 128, 3, 1, padding=1),  # 112, 112, 128
#             nn.ReLU(True),
#             nn.Conv2d(128, 128, 3, 1, padding=1),  # 112, 112, 128
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # 56, 56, 128
#             nn.Conv2d(128, 256, 3, 1, padding=1),  # 56, 56, 256
#             nn.ReLU(True),
#             nn.Conv2d(256, 256, 3, 1, padding=1),  # 56, 56, 256
#             nn.ReLU(True),
#             nn.Conv2d(256, 256, 3, 1, padding=1),  # 56, 56, 256
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # 28, 28, 256
#             nn.ReLU(True),
#             nn.Conv2d(256, 512, 3, 1, padding=1),  # 28, 28, 512
#             nn.ReLU(True),
#             nn.Conv2d(512, 512, 3, 1, padding=1),  # 28, 28, 512
#             nn.ReLU(True),
#             nn.Conv2d(512, 512, 3, 1, padding=1),  # 28, 28, 512
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # 14, 14, 512
#             nn.Conv2d(512, 512, 3, 1, padding=1),  # 14, 14, 512
#             nn.ReLU(True),
#             nn.Conv2d(512, 512, 3, 1, padding=1),  # 14, 14, 512
#             nn.ReLU(True),
#             nn.Conv2d(512, 512, 3, 1, padding=1),  # 14, 14, 512
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # 7, 7, 512
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(7 * 7 * 512, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 1000),
#         )
#
#         # self._initlize_weights()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         out = self.classifier(x)
#         return out


# # 附AlexNet的定义
# class Net(nn.Module):
#     def __init__(self): # 默认为两类，猫和狗
# #         super().__init__() # python3
#         super(Net, self).__init__()
#         # 开始构建AlexNet网络模型，5层卷积，3层全连接层
#         # 5层卷积层
#         self.conv1 = nn.Sequential(
#             # nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             # LRN(local_size=5, bias=1, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, groups=2, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             # LRN(local_size=5, bias=1, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
#         # 3层全连接层
#         # 前向计算的时候，最开始输入需要进行view操作，将3D的tensor变为1D
#         self.fc6 = nn.Sequential(
#             nn.Linear(in_features=6*6*256, out_features=4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout()
#         )
#         self.fc7 = nn.Sequential(
#             nn.Linear(in_features=4096, out_features=4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout()
#         )
#         self.fc8 = nn.Linear(in_features=4096, out_features=1)
#
#     def forward(self, x):
#         x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
#         x = x.view(-1, 6*6*256)
#         x = self.fc8(self.fc7(self.fc6(x)))
#         return x