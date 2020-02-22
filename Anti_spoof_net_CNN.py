import torch
import torch.nn as nn
import numpy as np
import torchvision

class block(nn.Module):

    def __init__(self,begin):
        super(block,self).__init__()

        if begin==True:
            self.cnn1=nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3,stride=1,padding=1)
        else:
            self.cnn1=nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=1)
        nn.init.xavier_normal(self.cnn1.weight) 
        self.bn1=nn.BatchNorm2d(128)
        self.non_linearity1=nn.CELU(alpha=1.0, inplace=False)
        
        self.cnn2=nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3,stride=1,padding=1)
        nn.init.xavier_normal(self.cnn2.weight)
        self.bn2=nn.BatchNorm2d(196)
        self.non_linearity2=nn.CELU(alpha=1.0, inplace=False)
        
        self.cnn3=nn.Conv2d(in_channels=196, out_channels=128, kernel_size=3,stride=1,padding=1)
        nn.init.xavier_normal(self.cnn3.weight)
        self.bn3=nn.BatchNorm2d(128)
        self.non_linearity3=nn.CELU(alpha=1.0, inplace=False)
        
        self.pool=nn.MaxPool2d(kernel_size=2)

    def forward(self,x,begin):
        
        x=self.cnn1(x)
        x=self.bn1(x)
        x=self.non_linearity1(x)
        x=self.cnn2(x)
        x=self.bn2(x)
        x=self.non_linearity2(x)
        x=self.cnn3(x)
        x=self.bn3(x)
        x=self.non_linearity3(x)
        x=self.pool(x)
    

class Anti_spoof_net_CNN(nn.Module):
  
    def __init__ (self):
        super(Anti_spoof_net_CNN,self).__init__()
            
        self.resize_32 = nn.Upsample(size=32, mode='nearest')
        self.resize_64 = nn.Upsample(size=64, mode='nearest')

        self.cnn0=nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3,stride=1,padding=1)
        nn.init.xavier_normal(self.cnn0.weight) 
        self.bn0=nn.BatchNorm2d(64)
        self.non_linearity0=nn.CELU(alpha=1.0, inplace=False)

        self.block1=block(True)
        self.block2=block(False)
        self.block3=block(False)
        
        #Feature map:
        self.cnn4=nn.Conv2d(in_channels=384,out_channels=128, kernel_size=3,stride=1,padding=1)
        self.cnn5=nn.Conv2d(in_channels=128,out_channels=3, kernel_size=3,stride=1,padding=1)
        self.cnn6=nn.Conv2d(in_channels=3,out_channels=1, kernel_size=3,stride=1,padding=1)
        
        #Depth map:
        self.cnn7=nn.Conv2d(in_channels=384,out_channels=128, kernel_size=3,stride=1,padding=1)
        self.cnn8=nn.Conv2d(in_channels=128,out_channels=64, kernel_size=3,stride=1,padding=1)
        self.cnn9=nn.Conv2d(in_channels=64,out_channels=1, kernel_size=3,stride=1,padding=1)



    def forward(self,x):
        
        x=self.cnn0(x)
        x=self.bn0(x)
        x=self.non_linearity0(x)
        
        #Block1
        x=self.block1(x)
        X1=self.resize_64(x)
        
        #Block2
        x=self.block2(x)
        X2=x
        
        #Block3:
        x=self.block3(x)
        X3=self.resize_64(x)
        
        X=torch.cat((X1,X2,X3),1)
        
        #Feature map:
        T=self.cnn4(X)
        T=self.cnn5(T)
        T=self.cnn6(T)
        T=self.resize_32(T)
        
        #Depth map:
        D=self.cnn7(X)
        D=self.cnn8(D)
        D=self.cnn9(D)
        D=self.resize_32(D)
        
        return D,T