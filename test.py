import torch
from pathlib import Path
# from main import
# from main import loss_func
import numpy as np
from dataset import testDataLoader,Dataset,test_data, test_labels,testSignData,testDataLoader
from model import resnet50
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.metrics import r2_score
from sklearn.metrics import f1_score

model_dir = '/home/mengying/公共的/04_PhenocamCNNR/02_Allsites_ROI14453/ResNet_Allsites45_ROI/saved_models/checkpoint_250/Resnet50_All_ROI_itr_250_train_0.000762_tar_0.000600.pth'

net = resnet50()
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
    print(targets,pred)

    plt.subplots(figsize=(5,5))
    # rms = mean_squared_error(targets, pred)
    # rmse = sqrt(mean_squared_error(targets, pred))
    r2 = r2_score(targets, pred)
    # rms = round(rms, 3)
    # rmse = round(rmse, 3)
    r2 = round(r2, 3)

    plt.scatter(targets,pred,85)
    rmse = sqrt(mean_squared_error(targets, pred))
    r2 = r2_score(targets, pred)
    f1 = f1_score(targets, pred,average="micro")
    # rms = round(rms, 3)
    rmse = round(rmse, 3)
    r2 = round(r2, 3)
    f1 = round(f1, 3)
    print('R^2: {:.3f} \tRMSE: {:.3f} \tAccuracy: {:.3f}\tF1: {:.3f}'.format(r2, rmse, accuracy, f1))

    plt.title("Test Accuracy")
    plt.xlabel("targets")
    plt.ylabel("prediction")

    plt.text(10, 280, "R^2 = ", c="blue")
    plt.text(10, 270, "RMSE= ", c="blue")
    plt.text(10, 260, "F1 = ", c="blue")
    # plt.text(40, 220, "Accuracy=", c="blue",fontsize=6)
    plt.annotate(r2, xy=(50, 280), c="blue")
    plt.annotate(rmse, xy=(55, 270), c="blue")
    plt.annotate(f1, xy=(45, 260), c="blue")

    plt.savefig("test.png")
    plt.show()

test(net, testDataLoader,loss_func)

