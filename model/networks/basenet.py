import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel=1,stride=1,use_batchnorm=True):
        super(ConvBlock, self).__init__()

        self.use_batchnorm = use_batchnorm

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels, 
                              kernel_size=kernel, 
                              stride=stride,
                              padding=(kernel-1)//2, 
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
    
    def forward(self,x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        x = self.relu(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,channels,use_batchnorm=True):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(channels,channels//2,kernel=1,stride=1,use_batchnorm=use_batchnorm)
        self.conv2 = ConvBlock(channels//2,channels,kernel=3,stride=1,use_batchnorm=use_batchnorm)
    
    def forward(self,x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        return res + x

class BaseNet(nn.Module):
    def __init__(self,args):
        super(BaseNet, self).__init__()
        self.base_ch_dim = args.base_ch_dim
        self.use_res = args.use_resblock
        self.steps = args.conv_steps
        self.kernel_size = args.kernel_size
        
        if self.steps<2:raise ValueError("Steps should be larger than 2")
        
        self.step_in_dims = [1]+[self.base_ch_dim*(s+1) for s in range(self.steps-1)]
        self.step_out_dims = [self.base_ch_dim*(s+1) for s in range(self.steps)]
        
        self.feature_extractor = []
            
        for s in range(self.steps):
            self.feature_extractor.append(ConvBlock(self.step_in_dims[s],
                                                    self.step_out_dims[s],
                                                    kernel=self.kernel_size,
                                                    use_batchnorm=args.use_batchnorm))
            if s<2:
                self.feature_extractor.append(nn.MaxPool2d(2))
            if self.use_res:
                self.feature_extractor.append(ResBlock(self.step_out_dims[s],use_batchnorm=args.use_batchnorm))

        self.feature_extractor = nn.Sequential(*self.feature_extractor)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(7*7*self.base_ch_dim*self.steps, 10)
        
    def forward(self, x):
        fmap = self.feature_extractor(x)
        features = fmap.view(fmap.size(0),-1)
        features = self.dropout(features)
        logits = self.fc(features)
        return logits