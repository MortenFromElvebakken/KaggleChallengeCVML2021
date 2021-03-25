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
    def __init__(self, trainingImages, vggNet, path):
        self.training = trainingImages
        self.vggNet = vggNet
        self.epochs = 40
        self.path = path

    def traingan(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def weights_init(m):
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.vggNet = self.vggNet.apply(weights_init)
        self.vggNet.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(self.vggNet.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)  # torch.optim.Adam(self.vggNet.parameters(), lr=0.001, betas=(0.9,0.99))
        #optimizer = torch.optim.Adam(self.vggNet.parameters(), lr=0.005, betas=(0.9, 0.99))

        train_loss = 0.0
        i = 1
        for epoch in range(self.epochs):
            # Dataloader returns the batches
            RunningLoss = 0.0
            for batchOfSamples in tqdm(self.training):

                batchOfImages = batchOfSamples['image'].to(device)
                labels = batchOfSamples['label'].to(device)
                optimizer.zero_grad()

                outputs = self.vggNet(batchOfImages)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                RunningLoss += loss.item()
                if i % 20 == 19:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, RunningLoss / 20))
                    print("outputs " + str(torch.max(outputs.data, 1)))
                    print("labels " + str(labels))
                    RunningLoss = 0.0
                i = i+1
        #torch.save(self.vggNet.state_dict(), self.path)
        outputPath = r'C:\Users\Morten From\PycharmProjects\KaggleChallengeCVML2021\data\finishedModels\Vgg19_1.pth'
        torch.save(self.vggNet.state_dict(), outputPath)
        return self.vggNet

