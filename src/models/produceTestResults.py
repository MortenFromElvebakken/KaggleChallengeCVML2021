import torch
from torch import nn
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd

class testResultsInpainting():
    def __init__(self, testImages, vggNet, classes):
        self.vggNet = vggNet
        self.testImages = testImages
        self.classes = classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def runTest(self):


        arr_flat = []
        self.vggNet.eval()
        with torch.no_grad():
            for batchOfSamples in tqdm(self.testImages):
                allLabels = []
                batchOfImages = batchOfSamples['image'].to(self.device)
                #labels = batchOfSamples['label'].to(self.device)
                outputs = self.vggNet(batchOfImages)
                _, PredictedLabels = torch.max(outputs.data, 1)
                allLabels.append(PredictedLabels.cpu().numpy())
                allLabels = allLabels
                arr_flat = np.append(arr_flat, allLabels)



        outputPath = r'C:\Users\Morten From\PycharmProjects\KaggleChallengeCVML2021\data\Results\Vgg19Results.csv'
        arr_flat = arr_flat.astype(int)
        arr_flat = arr_flat.tolist()
        df = pd.DataFrame()


        arr_flat = [x + 1 for x in arr_flat]
        df['Label'] = arr_flat
        df.index+=1
        df.to_csv(outputPath,index_label='ID')