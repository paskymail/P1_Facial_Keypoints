## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # maxpool that uses a square window of kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)

        #Second layer
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(4, 4)

        #Third layer
        self.conv3 = nn.Conv2d(64, 128, 3)

        #Fourth layer
        self.conv4 = nn.Conv2d(128, 256, 3)

        #fully connected
        # size after filter 1 = 224-4=220
        # size after pool = 220/2=110 
        # size after filter 2 = 110-2 = 108
        # size after pool 2 = 108/4 = 26
        # size after filter 3 = 26-2 = 24
        # size after pool 3 = 24/2 = 12
        # size after filter 4 = 12-2 = 10
        # size after pool 4 = 10/2 = 5

        #size flat = 256*5*5
        self.fc1 = nn.Linear(256*5*5, 1000)

        # dropout with p=0.1
        self.conv_drop = nn.Dropout(p=0.1)
        
        # dropout with p=0.5
        self.fc_drop = nn.Dropout(p=0.5)

        
        #dense 2
        self.fc2 = nn.Linear(1000, 1000)

        #fully connected with output 2
        self.fc3 = nn.Linear(1000, 2)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # one activated conv layer
        #x = F.relu(self.conv1(x))

        x = self.pool(F.relu(self.conv1(x)))

        x = self.conv_drop(x)

        x = self.pool2(F.relu(self.conv2(x)))

        x = self.conv_drop(x)

        x = self.pool(F.relu(self.conv3(x)))

        x = self.conv_drop(x)

        x = self.pool(F.relu(self.conv4(x)))

        x = self.conv_drop(x)
        
        #flat
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = self.fc_drop(x)

        x = F.relu(self.fc2(x))

        x = self.fc_drop(x)

        x = F.relu(self.fc3(x))
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
