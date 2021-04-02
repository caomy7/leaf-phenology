import torch
from pathlib import Path
# from main import
# from main import loss_func
import numpy as np
from dataset import testDataLoader,Dataset,test_data, test_labels,testSignData,testDataLoader
from model import Net
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.metrics import r2_score
from sklearn.metrics import f1_score

model_dir = '/home/mengying/公共的/04_PhenocamCNNR/01_Allsites_Images14453/AlexNet_Allsites/01_ResNet50/saved_models/checkpoint_370/Resnet50_All_ROI_itr_370_train_0.008778_tar_0.007821.pth'
# model_dir = '/home/mengying/公共的/04_PhenocamCNNR/02_Allsites_ROI14453/AlexNet_Allsites_ROI/saved_models/checkpoint_480/AlexNet_Dukehw_ROI_480_train_0.007667_tar_0.006534.pth'
test_dir = "./saved_test_epoch380/"
net = Net()
loss_func = torch.nn.MSELoss()

# def load_checkpoint(checkpoint, model, optimizer=None, map_location=None):
#     if not checkpoint.exists():
#         raise FileNotFoundError(f"File doesn't exist {checkpoint}")
#     state = torch.load(checkpoint, map_location=map_location)
#     model.load_state_dict(state['state_dict'])
#
#     if optimizer:
#         optimizer.load_state_dict(state['optim_dict'])
#     return state

def test(net, testDataLoader,loss_func):
    net.load_state_dict(torch.load(model_dir))
    print(net)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for sample in testDataLoader:

            inputs, targets = sample['X'], sample['Y'].long()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            _, pred = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()

            # pred = np.array(pred.cpu())
            pred = pred.cpu()
            accuracy = 100 * correct / total

            # np.savetxt(r"D:\04_Study\02_pytorch_github\10_regression\03_Case\03_face_Age\07_face-cnn\pre_result.txt",pred,delimiter = " ",fmt = "%f")

    print('\nAccuracy of the network on test: %d %%' % (100 * correct / total))

    targets = targets.cpu()
    print(targets, pred)

    plt.subplots(figsize=(5, 5))

    plt.scatter(targets, pred, 25)
    rmse = sqrt(mean_squared_error(targets, pred))
    r2 = r2_score(targets, pred)
    f1 = f1_score(targets, pred, average="micro")
    MAE = mean_absolute_error(targets, pred)
    # rms = round(rms, 3)
    rmse = round(rmse, 3)
    r2 = round(r2, 3)
    f1 = round(f1, 3)
    mAE = round(MAE, 3)
    print('R^2: {:.3f} \tRMSE: {:.3f} \tMAE: {:.3f}'.format(r2, rmse, MAE))

    dt = {"targets": targets, "Prediction": pred}
    data = pd.DataFrame(data=dt)

    tests = test_dir
    isExists = os.path.exists(tests)
    # 判断结果
    if not isExists:
        os.makedirs(tests)
    # name = 'Epoch: {} \tR^2: {:.3f} \tRMSE: {:.3f} \tMAE: {:.3f}'.format(epoch, r2, rmse, MAE) + ".csv"
    name = 'R^2: {:.3f} \tRMSE: {:.3f} \tMAE: {:.3f}'.format(r2, rmse, MAE) + ".csv"

    data.to_csv(test_dir + name)


test(net, testDataLoader,loss_func)

