# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 05:13:58 2021

@author: User
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import math


class simpleModel(nn.Module):
    def __init__(self, dropout, imgSize, numChannels,outputSize):
        super().__init__()
        self.drop = dropout
      
        # Fully connected 1
        self.fc1 = nn.Linear(imgSize**2*numChannels, 200) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=self.drop)
        
        
        # final layer
        self.fcF = nn.Linear(200, outputSize) 
        
    def forward(self, x):
        #flatten
        x = x.reshape(x.size(0), -1)
        
        #Dense
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fcF(out)
        
        return out


class simpleCNNModel(nn.Module):
    def __init__(self, dropout, imgSize, numImageChannels,outputSize):
        super().__init__()
        self.drop = dropout
        
        # Convolution 1+2:
        self.cnn1 = nn.Conv2d(in_channels=numImageChannels, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.relu1 = nn.ReLU()
        self.cnn2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolution 3+4:
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        # Convolution 5+6:
        self.cnn5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.relu5 = nn.ReLU()
        self.cnn6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(16)
        self.relu6 = nn.ReLU()
        # Max pool 2
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        # outputsize
        outSize = math.ceil(imgSize / 8)
        
        # Fully connected 1
        self.fc1 = nn.Linear(outSize*outSize*16, 200) 
        self.relu5 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=self.drop)
        
        # final layer
        self.fcF = nn.Linear(200, outputSize) 
    
    
    def forward(self, x, y=0.0):
        # Convolution 1+2
        out = self.cnn1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.cnn2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool1(out)
        
        # Convolution 3+4
        out = self.cnn3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.cnn4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.maxpool2(out)
        
        # Convolution 5+6
        out = self.cnn5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.cnn6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out = self.maxpool3(out)
        

        #flatten
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        
        #Dense
        out = self.fc1(out)
        out = self.relu5(out)
        out = self.dropout1(out)
        out = self.fcF(out)

        return out