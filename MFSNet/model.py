import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np
from torch import nn
from torch.nn import init

class ConvBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters, stride):
        super(ConvBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,1,stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.Conv2d(in_channel, F3, 1, stride=stride, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(inplace=True)
        
    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X    

class IndentityBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters):
        super(IndentityBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)
        
    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters):
        super(ConvTransposeBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.ConvTranspose2d(in_channel,F1,kernel_size=2,stride=2, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.ConvTranspose2d(in_channel,F3,kernel_size=2,stride=2, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(inplace=True)
        
    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X    
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.wRes50 = wide_resnet50_2(pretrained=True)

        
    def forward(self, x):
        x = self.wRes50.conv1(x)
        x = self.wRes50.bn1(x)
        x = self.wRes50.relu(x)
        x = self.wRes50.maxpool(x)

        x = self.wRes50.layer1(x) # [1, 256, 64, 64]
        feature1 = x
        
        x = self.wRes50.layer2(x) # [1, 512, 32, 32]
        feature2 = x
        
        x = self.wRes50.layer3(x) # [1, 1024, 16, 16]
        feature3 = x
        
        return feature1, feature2, feature3 



class MFF(nn.Module):
    def __init__(self):
        super(MFF, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 2, padding = 1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 2, padding = 1),
                                     nn.BatchNorm2d(1024),
                                     nn.ReLU(inplace=True)
                                     )
        
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 2, padding = 1),
                                     nn.BatchNorm2d(1024),
                                     nn.ReLU(inplace=True)
                                     )
        
        self.merge =nn.Sequential(nn.Conv2d(in_channels = 3072, out_channels = 1024, kernel_size = 1, stride = 1, padding = 0),
                                  nn.BatchNorm2d(1024),
                                  nn.ReLU(inplace=True)
                                  )
        
        self.fd = nn.Sequential(ConvBlock(in_channel =1024, kernel_size = 3, filters=[512,512,2048], stride=2),
                                      IndentityBlock(in_channel=2048, kernel_size=3, filters=[512,512,2048]),
                                      IndentityBlock(in_channel=2048, kernel_size=3, filters=[512,512,2048])
                                      )
        
    def forward(self, x1, x2, x3):
        output = torch.cat((self.branch1(x1),self.branch2(x2),x3),dim=1) # [1, 3072, 16, 16]
        output = self.merge(output) # [1, 1024, 16, 16]
        # output = self.branch1(x1) + self.branch2(x2) + x3
        output = self.fd(output) # [1, 2048, 8, 8]
        return output
    

class CBAMLayer(nn.Module):
    def __init__(self, channel=2048, reduction=8, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
            
        )
 
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.layer = nn.Sequential(
            nn.Linear(2048, 2048),  # 输入层和输出层的维度相同
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048)
        )
    def forward(self, x):
        
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        x = x.transpose(1, 3)
        x = self.layer(x)
        x = x.transpose(1, 3)
        return x
   
            
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer3 = nn.Sequential(ConvTransposeBlock(in_channel=2048, kernel_size=3, filters=[512, 1024, 1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512,1024,1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512,1024,1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512,1024,1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512,1024,1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512,1024,1024]),
                                    )
        self.layer2 = nn.Sequential(ConvTransposeBlock(in_channel=1024, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512])
                                    )
        self.layer1 = nn.Sequential(ConvTransposeBlock(in_channel=512, kernel_size=3, filters=[128, 256, 256]),
                                    IndentityBlock(in_channel=256, kernel_size=3, filters=[128, 256, 256]),
                                    IndentityBlock(in_channel=256, kernel_size=3, filters=[128, 256, 256])
                                    )
        
        
    def forward(self, x):
        
        x = self.layer3(x) # [1, 1024, 16, 16]
        feature3 = x
        x = self.layer2(x) # [1, 512, 32, 32]
        feature2 = x
        x = self.layer1(x) # [1, 256, 64, 64]
        feature1 = x
        
        return feature1, feature2, feature3


class MemoryBank:
    def __init__(self):
        self.memory = []

    def add_memory(self,feature):
        self.memory.append(feature)
        
    def find_closest(self, x):
        
        # 初始化最小距离和对应的索引
        min_loss = float('inf')
        closest_index = -1
        
        # 使用均方误差损失函数
        loss_func = nn.MSELoss()
        
        # 计算 x 与 feature_list 中每个张量的距离，并找到最小距离的张量索引
        for i, feature in enumerate(self.memory):
            loss = loss_func(x, feature)
            if loss < min_loss:
                min_loss = loss
                closest_index = i
        
        # 返回距离最近的张量和对应的索引
        # print(closest_index)
        return self.memory[closest_index]
    
    def clear_memory(self):
        self.memory = []
    
    

