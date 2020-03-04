import torch
import torch.nn as nn
import numpy as np
import torchvision

class Anti_spoof_net_RNN(nn.Module):
  
    def __init__ (self):
        super(Anti_spoof_net_RNN,self).__init__()
        
        self.hidden_dim=100
        self.input_dim=32*32
        self.num_layers=1
        self.batch_size=1

        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim), torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        self.LSTM=nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,num_layers=self.num_layers)
        self.fc=nn.Linear(self.hidden_dim,2)
        
    def forward(self,F):
            
        #F est de dimension [5,32,32,1]
        F = F.view(5,1,-1)
        lstm_out, self.hidden = self.LSTM(F,self.hidden) #lstm_out[5,1,100]
        R = self.fc(lstm_out) #F[5,1,2]
        R = torch.fft(R, signal_ndim=1, normalized=False)   
        return R #F[5,1,2]
