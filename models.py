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

        # dropout with p=0.1
        self.conv_drop = nn.Dropout(p=0.1)

        #Second layer
        self.conv2 = nn.Conv2d(32, 64, 5)

        #Third layer
        self.conv3 = nn.Conv2d(64, 128, 4)

        #Fourth layer
        self.conv4 = nn.Conv2d(128, 256, 4)

        #Fifth layer
        self.conv5 = nn.Conv2d(256, 512, 2)


        #fully connected
        # size after filter 1 = 224-4=220
        # size after pool = 220/2=110 
        # size after filter 2 = 110-4 = 106
        # size after pool 2 = 106/2 = 53
        # size after filter 3 = 53-3 = 50
        # size after pool 3 = 50/2 = 25
        # size after filter 4 = 25-3 = 22
        # size after pool 4 = 22/2 = 11
        # size after filter 5 = 11-1 = 10
        # size after pool 5 = 10/2 = 5     

        #size flat = 256*5*5
        self.fc1 = nn.Linear(512*5*5, 6000)
        
        # dropout with p=0.5
        self.fc_drop = nn.Dropout(p=0.4)

        #dense 2
        self.fc2 = nn.Linear(6000, 1000)

        #dense 3
        self.fc3 = nn.Linear(1000, 1000)

        #dense 4
        self.fc4 = nn.Linear(1000, 500)

        #fully connected with output 68*2
        self.fc_out = nn.Linear(500, 68*2)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # one activated conv layer
        #x = F.relu(self.conv1(x))

        x = self.pool(F.relu(self.conv1(x)))

        x = self.conv_drop(x)

        x = self.pool(F.relu(self.conv2(x)))

        x = self.conv_drop(x)

        x = self.pool(F.relu(self.conv3(x)))

        x = self.conv_drop(x)

        x = self.pool(F.relu(self.conv4(x)))

        x = self.conv_drop(x)

        x = self.pool(F.relu(self.conv5(x)))

        x = self.conv_drop(x)
        
        #flat
        x = x.view(x.size(0), -1)

        x = F.elu(self.fc1(x))

        x = self.fc_drop(x)

        x = F.elu(self.fc2(x))

        x = self.fc_drop(x)

        x = F.elu(self.fc3(x))

        x = self.fc_drop(x)

        x = F.elu(self.fc4(x))

        x = self.fc_drop(x)

        x = self.fc_out(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
