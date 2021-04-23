import torch
from torch import nn
import torchvision

#Der findes indbygget vgg nets, men vi bygger vores eget og træner det
class VGGnet1ConvLayer(nn.Module):
    def __init__(self, in_size, out_size, kernels, normalize=True):
        super(VGGnet1ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_size,out_size,kernels,stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchNorm = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x

class VGGnet2ConvLayer(nn.Module):
    def __init__(self, in_size, out_size, kernels):
        super(VGGnet2ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_size,out_size,kernels,stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_size,out_size,kernels,stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchNorm = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)
        return x

class VGGnet4ConvLayer(nn.Module):
    def __init__(self, in_size, out_size, kernels):
        super(VGGnet4ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_size,out_size,kernels,stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_size,out_size,kernels,stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_size, out_size, kernels, stride=1, padding=1)
        self.conv4 = nn.Conv2d(out_size, out_size, kernels, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchNorm = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.relu(x)
        return x

class VGGnetFullyConnected(nn.Module):
    def __init__(self, in_size, out_size,final_size):
        super(VGGnetFullyConnected, self).__init__()
        self.fc1 = nn.Linear(in_size*8*8,out_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(out_size, out_size)
        self.fc3 = nn.Linear(out_size, final_size)
        self.relu = nn.ReLU(True)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class Vgg19(nn.Module):
    def __init__(self, batch_size=4):
        super(Vgg19, self).__init__()
        self.batchSize = batch_size
        self.first = VGGnet2ConvLayer(3, 64, 3)
        self.second = VGGnet2ConvLayer(64, 128, 3)
        self.third = VGGnet4ConvLayer(128, 264, 3)
        self.fourth = VGGnet4ConvLayer(264, 512, 3)
        self.fifth = VGGnet4ConvLayer(512, 512, 3)
        self.sixth = VGGnetFullyConnected(512, 4096, 29)

    def forward(self, input):
        x1 = self.first(input)
        x2 = self.second(x1)
        x3 = self.third(x2)
        x4 = self.fourth(x3)
        x5 = self.fifth(x4)
        x5 = x5.view(self.batchSize, -1)
        x6 = self.sixth(x5)
        return x6

class Vgg11(nn.Module):
    def __init__(self, batch_size=4):
        super (Vgg11, self).__init__()
        self.batchSize = batch_size
        self.first = VGGnet1ConvLayer(3, 64, 3)
        self.second = VGGnet1ConvLayer(64, 128, 3)
        self.third = VGGnet2ConvLayer(128, 264, 3)
        self.fourth = VGGnet2ConvLayer(264, 512, 3)
        self.fifth = VGGnet2ConvLayer(512, 512, 3)
        self.sixth = VGGnetFullyConnected(512, 4096, 29)

    def forward(self, input):
        x1 = self.first(input)
        x2 = self.second(x1)
        x3 = self.third(x2)
        x4 = self.fourth(x3)
        x5 = self.fifth(x4)
        x5 = x5.view(self.batchSize, -1)  # Ret til batch size tal hvis vi ændrer batch size
        x6 = self.sixth(x5)
        # x6 = nn.Softmax(x6)
        return x6

