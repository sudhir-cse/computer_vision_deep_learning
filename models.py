## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def conv_layer(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels)
    )

def linear_layer(in_features, out_features, drop_p):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.Dropout(drop_p)
    )

class Net(nn.Module):

    def __init__(self, init_weights=True):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # Input image size (224x224)
        # Convolutional layers        
        self.conv_layer_1 = conv_layer(1, 32, 4)     # Conv: (32 X 221 X 221), Pool: (32 X 110 X 110)
        self.conv_layer_2 = conv_layer(32, 64, 3)    # Conv: (64 X 108 X 108), Pool: (64 X 54 X 54) 
        self.conv_layer_3 = conv_layer(64, 128, 3)   # Conv: (128 X 52 X 52), Pool: (128 X 26 X 26)
        self.conv_layer_4 = conv_layer(128, 256, 3)  # Conv: (256 X 24 X 24), Pool: (256 X 12 X 12)
        self.conv_layer_5 = conv_layer(256, 512, 3)  # Conv: (512 X 10 X 10), Pool: (512 X 5 X 5)

        # Linear layer
        self.linear_layer_1 = linear_layer(512*5*5, 4000, 0.25)
        self.linear_layer_2 = linear_layer(4000, 2000, 0.25)
        self.linear_layer_3 = nn.Linear(2000, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        if init_weights:
            self._init_weights()
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # conv and pooling layers
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.conv_layer_5(x)
        
        # flatten the input image 
        x = x.view(x.size(0), -1)
        
        # Linear layer
        x = self.linear_layer_1(x)
        x = self.linear_layer_2(x)
        x = self.linear_layer_3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    # Initialize model weithts
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                I.uniform_(m.weight)
                if m.bias is not None:
                    I.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                I.xavier_uniform_(m.weight)
                I.constant_(m.bias, 0)
    
