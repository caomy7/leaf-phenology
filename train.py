import torch
from pathlib import Path
from dataset import *
import time
import datetime
import os
import pandas as pd
import matplotlib

from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
import time

import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
epochs = 500

Dates = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())

model_dir = "./saved_models/"
test_dir = "./test_result/"

def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)

def train(epoch, net, trainDataLoader, optimizer, criterion, validDataLoader,testDataLoader):
    net.train()
    train_loss = 0
    number = 0
    epoch_st = time.time()
    correct, total = 0, 0
    for sample in trainDataLoader:

        inputs, targets = sample['X'], sample['Y'].float()
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
            # print(inputs.shape,targets.shape)

        optimizer.zero_grad()
        outputs = net(inputs)
        # print(outputs.shape,targets.shape)

        loss = criterion(outputs[:,0], targets)
        loss.backward()
        optimizer.step()
        # number += 1
        train_loss += loss.item()
    # train_loss = train_loss/number
    if epoch % 10 == 0:
    # if epoch % 50 == 0:


        net.eval()
        valid_loss = 0
        number = 0
        with torch.no_grad():
            for sample in validDataLoader:

                inputs, targets = sample['X'], sample['Y'].float()
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()
                    print(targets.shape, outputs.shape)

                outputs = net(inputs)

                outputs[outputs<0] = 0


                loss = criterion(outputs[:, 0], targets)
                valid_loss += loss.item()
                # _, pred = torch.max(outputs, 1)
                pred = outputs
                # print(targets.shape, pred.shape)

                # calculate average losses
                train_loss = train_loss / len(trainDataLoader.sampler)
                valid_loss = valid_loss / len(validDataLoader.sampler)
                total += targets.size(0)
                correct += (pred == targets).sum().item()
                accuracy = 100 * correct / total

                print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAccuracy: {:.6f}'.format(epoch,
                                                                                                              train_loss,
                                                                                                              valid_loss,
                                                                                                              accuracy))
                # print(targets.shape, pred.shape)
                target = targets.cpu().detach()
                pre = pred.cpu().detach()
                # print(target.shape,pre.shape)

                rmse = sqrt(mean_squared_error(target, pre))
                r2 = r2_score(target, pre)
                # f1 = f1_score(target, pre, average="micro")

                mae = mean_absolute_error(target, pre)


        print('Epoch: {} \tR^2: {:.3f} \tRMSE: {:.3f} \tmae: {:.3f}'.format(epoch, r2, rmse, mae))


        train_result = 'Epoch: {} \tR^2: {:.3f} \tRMSE: {:.3f} \tMAE: {:.3f}'.format(epoch, r2, rmse, mae)

        train_file_name = "train_result"

        with open (train_file_name,"a+") as f:
            f.writelines(train_result)

        checkpoints = model_dir + "checkpoint_" + str(epoch)
        isExists = os.path.exists(checkpoints)
        # 判断结果
        if not isExists:
            os.makedirs(checkpoints)

        if epoch > 50:  # save model every 2000 iterations
            print("saving*********************************************************")

            torch.save(net.state_dict(),checkpoints + "/Resnet50_All_ROI_itr_%d_train_%3f_tar_%3f.pth" % (epoch, train_loss, valid_loss))

        epoch_time = time.time() - epoch_st
        remain_time = epoch_time * (epochs - 1 - epoch)
        m, s = divmod(remain_time, 60)
        h, m = divmod(m, 60)
        if s != 0:
            train_time = "Remaining training time = %d hours %d minutes %d seconds\n" % (h, m, s)
        else:
            train_time = "Remaining training time : Training completed.\n"

        LOG(train_time)

    with torch.no_grad():
        for sample in testDataLoader:

            inputs, targets = sample['X'], sample['Y'].float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            pred = outputs

            outputs[outputs < 0] = 0
            outputs[outputs < 0] = 0

            total += targets.size(0)

            pred = pred.cpu().detach()
            targets = targets.cpu().detach()
            print(targets, pred)


            rmse = sqrt(mean_squared_error(targets, pred))
            r2 = r2_score(targets, pred)

            MAE = mean_absolute_error(targets, pred)
            # rms = round(rms, 3)
            rmse = round(rmse, 3)
            r2 = round(r2, 3)
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
        name = 'R^2{:.3f} RMSE{:.3f} MAE{:.3f}'.format(r2, rmse, MAE) + ".csv"
        data.to_csv(test_dir + name)








