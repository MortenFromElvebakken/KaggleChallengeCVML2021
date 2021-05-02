import torch
from torch import nn
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import glob
from pathlib import Path


class trainInpainting():
    def __init__(self, trainingImages, validImages, vggNet, path, epochs):
        self.training = trainingImages
        self.valid = validImages
        self.vggNet = vggNet
        self.epochs = epochs
        self.path = path

    def traingan(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # self.vggNet = self.vggNet.apply(weights_init)
        self.vggNet.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(self.vggNet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4,
                                    nesterov=True)  # torch.optim.Adam(self.vggNet.parameters(), lr=0.001, betas=(0.9,0.99))
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        # optimizer = torch.optim.Adam(self.vggNet.parameters(), lr=0.005, betas=(0.9, 0.99))

        runningCounter = 0
        for epoch in range(self.epochs):
            # Dataloader returns the batches
            train_loss_running = 0.0
            valid_loss_running = 0.0
            print("epoch started:" + str(epoch))
            for batchOfSamples in tqdm(self.training, leave=True, disable=True):

                batchOfImages = batchOfSamples['image'].to(device)
                labels = batchOfSamples['label'].to(device)
                optimizer.zero_grad()

                outputs = self.vggNet(batchOfImages)
                loss = criterion(outputs, labels)
                loss.backward()
                train_loss_running = train_loss_running + loss
                optimizer.step()
                if epoch < 150:
                    lr = 1e-1
                if epoch == 150:
                    lr = 1e-2
                if epoch == 200:
                    lr = 1e-3
                if epoch == 250:
                    lr = 1e-4
                if epoch == 300:
                    lr = 1e-5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            #Indsæt validation check her
            #laver en der tæller op, og hvis valid loss har været højere end training loss 5 epoker i streg,
            #så gemmer den model og breaker loop
            with torch.no_grad():
                for batchOfSamples in tqdm(self.valid, leave=True, disable=True):
                    batchOfImages = batchOfSamples['image'].to(device)
                    labels = batchOfSamples['label'].to(device)
                    optimizer.zero_grad()
                    outputs = self.vggNet(batchOfImages)
                    validLoss = criterion(outputs,labels)
                    valid_loss_running = valid_loss_running + validLoss

                epoch_loss_train = train_loss_running / len(self.training)
                epoch_loss_val = valid_loss_running / len(self.valid)
                if epoch_loss_train < epoch_loss_val:
                    runningCounter = 0
                else:
                    runningCounter = runningCounter + 1
                if runningCounter == 4:
                    break

        outputPath = r'C:\Users\Morten From\PycharmProjects\KaggleChallengeCVML2021\data\finishedModels\DenseNet161_ValidLossBreak.pth'
        torch.save(self.vggNet.state_dict(), outputPath)
        return self.vggNet