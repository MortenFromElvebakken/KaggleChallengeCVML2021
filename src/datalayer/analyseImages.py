import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import albumentations as A
import pandas as pd
import numpy as np
import cv2
import os


#inspireret fra https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?fbclid=IwAR3UzylpXP1ob0MZd-Ic3BZZKfs0zIcgqaxGl6qtjqw6M3F05V1ufpmW5j8
class AnalyzeDataSet():
    def __init__(self, path):
        self.path = path

        data_list = []
        for file in os.listdir(self.path):
            if file.endswith(".jpg"):
                pathToFile = os.path.join(self.path, file)
                data_list.append(pathToFile)

        df = []
        for i in range(len(data_list)):
            image = cv2.imread(data_list[i])

            r, g, b = cv2.split(image)
            b = b.flatten()
            r = r.flatten()
            g = g.flatten()
            df.append(pd.DataFrame(np.stack([r, g, b], axis=1), columns=['Red', 'Green', 'Blue']))
        df_merged = pd.concat(df)
        self.getMedian(df_merged, "TrainSet")
        self.getMean(df_merged, "TrainSet")
        self.getstd(df_merged, "TrainSet")

    def getMedian(self, df, name):
            Red_median = df['Red'].median()
            Gren_median = df['Green'].median()
            Blue_median = df['Blue'].median()
            print("Red median for " + str(name) + " is " + str(round(Red_median, 2)))
            print("Green median for " + str(name) + " is " + str(round(Gren_median, 2)))
            print("Blue median for " + str(name) + " is " + str(round(Blue_median, 2)))

    def getMean(self, df, name):
            Red_median = df['Red'].mean()
            Gren_median = df['Green'].mean()
            Blue_median = df['Blue'].mean()
            print("Red mean for " + str(name) + " is " + str(round(Red_median, 2)))
            print("Green mean for " + str(name) + " is " + str(round(Gren_median, 2)))
            print("Blue mean for " + str(name) + " is " + str(round(Blue_median, 2)))

    def getstd(self, df, name):
            Red_median = df['Red'].std()
            Gren_median = df['Green'].std()
            Blue_median = df['Blue'].std()
            print("Red std for " + str(name) + " is " + str(round(Red_median, 2)))
            print("Green std for " + str(name) + " is " + str(round(Gren_median, 2)))
            print("Blue std for " + str(name) + " is " + str(round(Blue_median, 2)))