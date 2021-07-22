
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision
import torch
from torchvision import models, datasets

class CRNN_Base(nn.Module):
    def __init__(self, class_num, c, h, w, k, filters, poolings, dropout_rate, gru_dropout=0.3, gru_units=32):
        super(CRNN_Base, self).__init__()
        input_shape = (c, h, w)
        # CNN
        self.bn0 = nn.BatchNorm2d(num_features=c)
        self.pad1 = nn.ZeroPad2d((int(k/2), int(k/2), int(k/2), int(k/2)))
        self.conv1 = nn.Conv2d(c, filters[0], kernel_size=k, stride=1)
        self.act1 = nn.ELU()
        self.bn1 = nn.BatchNorm2d(num_features=filters[0])
        self.maxPool1 = nn.MaxPool2d(kernel_size=poolings[0], stride=poolings[0])
        self.drouput1 = nn.Dropout2d(dropout_rate)

        self.pad2 = nn.ZeroPad2d((int(k/2), int(k/2), int(k/2), int(k/2)))
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=k)
        self.act2 = nn.ELU()
        self.bn2 = nn.BatchNorm2d(num_features=filters[1])
        self.maxPool2 = nn.MaxPool2d(kernel_size=poolings[1], stride=poolings[1])
        self.drouput2 = nn.Dropout2d(dropout_rate)

        self.pad3 = nn.ZeroPad2d((int(k/2), int(k/2), int(k/2), int(k/2)))
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=k)
        self.act3 = nn.ELU()
        self.bn3 = nn.BatchNorm2d(num_features=filters[2])
        self.maxPool3 = nn.MaxPool2d(kernel_size=poolings[2], stride=poolings[2])
        self.drouput3 = nn.Dropout2d(dropout_rate)

        self.pad4 = nn.ZeroPad2d((int(k/2), int(k/2), int(k/2), int(k/2)))
        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_size=k)
        self.act4 = nn.ELU()
        self.bn4 = nn.BatchNorm2d(num_features=filters[3])
        self.maxPool4 = nn.MaxPool2d(kernel_size=poolings[3],stride=poolings[3])
        self.drouput4 = nn.Dropout2d(dropout_rate)
        # Output is (m, chan, freq, time) -> Needs to be reshaped for feeding to GRU units
        # We will handle the reshape in the forward method

        # RNN
        self.gru = nn.GRU(input_size=256, hidden_size=32, batch_first=True, num_layers=2, dropout=gru_dropout)
        #self.gru2 = nn.GRU(input_size=32, hidden_size=32, batch_first=True, dropout=gru_dropout)

        # Dense and softmax
        self.dense1 = nn.Linear(32, class_num)
        self.softm = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # CNN forward
        x = self.bn0(x)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.maxPool1(x)
        x = self.drouput1(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.maxPool2(x)
        x = self.drouput2(x)

        x = self.pad3(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x = self.maxPool3(x)
        x = self.drouput3(x)

        x = self.pad4(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.bn4(x)
        x = self.maxPool4(x)
        x = self.drouput4(x)

        # Reshape
        x = x.permute(0,3,2,1)
        x = torch.reshape(x, (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]*x.shape[3])))
        # RNN forward
        x = self.gru(x)[1][0]
        # Dense and softmax forward
        x = self.dense1(x)
        x = self.softm(x)

        return x


class CRNN_Larger(nn.Module):
    def __init__(self, class_num, c, h, w, k, filters, poolings, dropout_rate, gru_dropout=0.3, gru_units=32):
        super(CRNN_Larger, self).__init__()
        input_shape = (c, h, w)
        # CNN
        self.bn0 = nn.BatchNorm2d(num_features=c)
        self.pad1 = nn.ZeroPad2d((int(k/2), int(k/2), int(k/2), int(k/2)))
        self.conv1 = nn.Conv2d(c, filters[0], kernel_size=k, stride=1)
        self.act1 = nn.ELU()
        self.bn1 = nn.BatchNorm2d(num_features=filters[0])
        self.maxPool1 = nn.MaxPool2d(kernel_size=poolings[0], stride=poolings[0])
        self.drouput1 = nn.Dropout2d(dropout_rate)

        self.pad2 = nn.ZeroPad2d((int(k/2), int(k/2), int(k/2), int(k/2)))
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=k)
        self.act2 = nn.ELU()
        self.bn2 = nn.BatchNorm2d(num_features=filters[1])
        self.maxPool2 = nn.MaxPool2d(kernel_size=poolings[1], stride=poolings[1])
        self.drouput2 = nn.Dropout2d(dropout_rate)

        self.pad3 = nn.ZeroPad2d((int(k/2), int(k/2), int(k/2), int(k/2)))
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=k)
        self.act3 = nn.ELU()
        self.bn3 = nn.BatchNorm2d(num_features=filters[2])
        self.maxPool3 = nn.MaxPool2d(kernel_size=poolings[2], stride=poolings[2])
        self.drouput3 = nn.Dropout2d(dropout_rate)

        self.pad4 = nn.ZeroPad2d((int(k/2), int(k/2), int(k/2), int(k/2)))
        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_size=k)
        self.act4 = nn.ELU()
        self.bn4 = nn.BatchNorm2d(num_features=filters[3])
        self.maxPool4 = nn.MaxPool2d(kernel_size=poolings[3],stride=poolings[3])
        self.drouput4 = nn.Dropout2d(dropout_rate)

        self.pad5 = nn.ZeroPad2d((int(k/2), int(k/2), int(k/2), int(k/2)))
        self.conv5 = nn.Conv2d(filters[3], filters[4], kernel_size=k)
        self.act5 = nn.ELU()
        self.bn5 = nn.BatchNorm2d(num_features=filters[4])
        self.maxPool5 = nn.MaxPool2d(kernel_size=poolings[4],stride=poolings[4])
        self.drouput5 = nn.Dropout2d(dropout_rate)
        # Output is (m, chan, freq, time) -> Needs to be reshaped for feeding to GRU units
        # We will handle the reshape in the forward method

        # RNN
        self.gru = nn.GRU(input_size=1024, hidden_size=32, batch_first=True, num_layers=2, dropout=gru_dropout)

        # Dense and softmax
        self.dense1 = nn.Linear(32, class_num)
        self.softm = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # CNN forward
        x = self.bn0(x)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.maxPool1(x)
        x = self.drouput1(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.maxPool2(x)
        x = self.drouput2(x)

        x = self.pad3(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x = self.maxPool3(x)
        x = self.drouput3(x)

        x = self.pad4(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.bn4(x)
        x = self.maxPool4(x)
        x = self.drouput4(x)

        x = self.pad5(x)
        x = self.conv5(x)
        x = self.act5(x)
        x = self.bn5(x)
        x = self.maxPool5(x)
        x = self.drouput5(x)

        # Reshape
        x = x.permute(0,3,2,1)
        x = torch.reshape(x, (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]*x.shape[3])))
        # RNN forward
        x = self.gru(x)[1][0]
        # Dense and softmax forward
        x = self.dense1(x)
        x = self.softm(x)
        return x

class CRNN_ResNet18(nn.Module):
    def __init__(self, class_num, c, h, w, k, filters, poolings, dropout_rate, gru_dropout=0.3, gru_units=32):
        # Backbone
        super(CRNN_ResNet18, self).__init__()
        input_shape = (c, h, w)
        self.backbone = torchvision.models.resnet18(pretrained=True)
        modules = list(self.backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        ct = 0
        for child in self.backbone.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False
        
        # RNN
        self.gru = nn.GRU(input_size=512, hidden_size=32, batch_first=True, num_layers=3, dropout=gru_dropout)
        #self.gru2 = nn.GRU(input_size=32, hidden_size=32, batch_first=True, dropout=gru_dropout)

        # Dense and softmax
        self.dense1 = nn.Linear(32, class_num)
        self.softm = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # Backbone forward
        x = self.backbone(x)

        # Reshape
        x = x.permute(0,3,2,1)
        x = torch.reshape(x, (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]*x.shape[3])))

        # RNN forward
        x = self.gru(x)[1][0]

        # Dense and softmax forward
        x = self.dense1(x)
        x = self.softm(x)
        return x