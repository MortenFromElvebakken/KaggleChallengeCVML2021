import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from natsort import natsorted
import albumentations as A
import pandas as pd
import numpy as np
import cv2
import os



#inspireret fra https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?fbclid=IwAR3UzylpXP1ob0MZd-Ic3BZZKfs0zIcgqaxGl6qtjqw6M3F05V1ufpmW5j8
#inspireret fra https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?fbclid=IwAR3UzylpXP1ob0MZd-Ic3BZZKfs0zIcgqaxGl6qtjqw6M3F05V1ufpmW5j8
class CreateDataloaders():
    def __init__(self, normalize=True, batch_size=4, num_workers=2):
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        #Remove .transforms if on collab.
        train_transform = A.Compose([
            A.transforms.HorizontalFlip(p=0.5),
            A.transforms.VerticalFlip(p=0.5),
            A.transforms.Normalize((0.57, 0.94, 0.45), (0.15, 0.17, 0.10))
        ]) #RGB mean ( 0.57,0.94,0.45) #rgb STD (0.15,0.17,0.10)
        test_transform = A.Compose([
            A.transforms.Normalize((0.57, 0.94, 0.45), (0.15, 0.17, 0.10))
            #A.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        TrainDir = r'C:\Users\Morten From\PycharmProjects\KaggleChallengeCVML2021\data\Train'
        ValDir = r'C:\Users\Morten From\PycharmProjects\KaggleChallengeCVML2021\data\Validation'
        #TestDirImg = r'C:\Users\Morten From\PycharmProjects\KaggleChallengeCVML2021\data\Test\ExtraTest'
        TestDirImg = r'C:\Users\Morten From\PycharmProjects\KaggleChallengeCVML2021\data\Test\TestImages'

        #CoLab
        #TrainDir = r'/content/drive/MyDrive/Colab Notebooks/Dataset/Train'
        #ValDir = r'/content/drive/MyDrive/Colab Notebooks/Dataset/Validation'
        #TestDirImg = r'/content/drive/MyDrive/Colab Notebooks/Dataset/Test/TestImages'

        #Server poly on AU
        #TrainDir = r'/workspace/CV_Jacob/Kaggle_Challenge_Computer_Vision/Data/Train'
        #ValDir = r'/workspace/CV_Jacob/Kaggle_Challenge_Computer_Vision/Data/Validation'
        #TestDirImg = r'/workspace/CV_Jacob/Kaggle_Challenge_Computer_Vision/Data/Test/TestImages'

        TrainDirImg = str(TrainDir + '/TrainImages')
        ValDirImg = str(ValDir + '/ValidationImages')
        TrainLlbs = str(TrainDir + '/trainLbls.csv')
        ValLbls = str(ValDir + '/valLbls.csv')



        list_train_data = natsorted([os.path.join(TrainDirImg, f) for f in os.listdir(TrainDirImg)])
        list_test_data = natsorted([os.path.join(TestDirImg, f) for f in os.listdir(TestDirImg)])
        list_val_data = natsorted([os.path.join(ValDirImg, f) for f in os.listdir(ValDirImg)])


        train_data = AlbumentationImageDataset(image_list=list_train_data, csv_file=TrainLlbs, transform=train_transform)
        test_data = AlbumentationImageDataset(image_list=list_test_data, csv_file=None, transform=test_transform)
        val_data = AlbumentationImageDataset(image_list=list_val_data, csv_file=ValLbls, transform=test_transform)

        test123 = pd.read_csv(TrainLlbs, header=None, names=['Labels'])
        self.classes = test123['Labels'].unique()

        weights = self.make_weights_for_balanced_classes(train_data.lbls.Labels-1  ,len(self.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        self.train_data_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler,
                                                        drop_last=True, pin_memory=True) #,shuffle = true
        self.test_data_loader = torch.utils.data.DataLoader(test_data,
                                                       batch_size=self.batch_size,
                                                       shuffle=False, num_workers=self.num_workers,
                                                       drop_last=True, pin_memory=True)

        self.val_data_loader = torch.utils.data.DataLoader(val_data,
                                                            batch_size=self.batch_size,
                                                            shuffle=True, num_workers=self.num_workers,
                                                            drop_last=True, pin_memory=True)



    def getDataloaders(self):
        return self.train_data_loader, self.test_data_loader, self.val_data_loader, self.classes

    def make_weights_for_balanced_classes(self, images, nclasses):
        count = [0] * nclasses
        for item in images:
            count[item] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val]
        return weight

class AlbumentationImageDataset(Dataset):
    def __init__(self, image_list, csv_file =None, transform=None):
        self.image_list = image_list
        self.transform = transform
        if csv_file is not None:
            self.lbls = pd.read_csv(csv_file, header=None, names=['Labels'])
        else:
            self.lbls = None

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, i):
        image = cv2.imread(self.image_list[i], -1)
        #Check if float
        if self.transform:
            image = self.transform(image=np.array(image))['image']
            #image = image/255
            image = np.float32(image)
            image = image.transpose(2, 0, 1)
        if self.lbls is not None:
            label = self.lbls.Labels[i]-1
            sample = {'image': image, 'label': label}
        else:
            sample = {'image': image}

        #image = torch.from_numpy(np.array(image).astype(np.float32)).transpose(0, 1).transpose(0, 2).contiguous() #Check rækkefølgen
        #sample = {'image': image, 'label': label}
        return sample