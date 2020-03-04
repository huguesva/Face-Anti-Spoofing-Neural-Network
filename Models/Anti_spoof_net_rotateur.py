import torch
import torch.nn as nn
import numpy as np
import torchvision

class Anti_spoof_net_rotateur(nn.Module):
  
    def __init__ (self):
        super(Anti_spoof_net_rotateur,self).__init__()
            
        self.resize_32 = nn.Upsample(size=32, mode='nearest')
        self.resize_64 = nn.Upsample(size=64, mode='nearest')

        self.cnn0=nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3,stride=1,padding=1)
        nn.init.xavier_normal(self.cnn0.weight) 
        self.bn0=nn.BatchNorm2d(32)
        self.non_linearity0=nn.CELU(alpha=1.0, inplace=False)

        self.cnn1=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3,stride=1,padding=1)
        nn.init.xavier_normal(self.cnn1.weight) 
        self.bn1=nn.BatchNorm2d(64)
        self.non_linearity1=nn.CELU(alpha=1.0, inplace=False)
        
        self.cnn2=nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3,stride=1,padding=1)
        nn.init.xavier_normal(self.cnn2.weight)
        self.bn2=nn.BatchNorm2d(1)
        self.non_linearity2=nn.CELU(alpha=1.0, inplace=False)

        self.fc=nn.Linear(32*32,8192)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
      
        
    def forward(self,x):
        
        x=self.cnn0(x)
        x=self.bn0(x)
        x=self.non_linearity0(x)
        x=self.pool(x)
        
        x=self.cnn1(x)
        x=self.bn1(x)
        x=self.non_linearity1(x)
        x=self.pool(x)

        x=self.cnn2(x)
        x=self.bn2(x)
        x=self.non_linearity2(x)
        x=self.pool(x)

        x=x.view(-1,32*32)

        x=self.fc(x)
    
        return x