import torch
import torch.nn as nn
import numpy as np
import torchvision
import Anti_spoof_net_CNN 
import Anti_spoof_net_RNN

class Anti_spoof_net(nn.Module):
  
    def __init__ (self):
        super(Anti_spoof_net,self).__init__()

        self.threshold = 0.1
        self.CNN = Anti_spoof_net_CNN.Anti_spoof_net_CNN()
        self.RNN = Anti_spoof_net_RNN.Anti_spoof_net_RNN()
        
    def forward(self,x,turned,anchors):
            
        D,T = self.CNN(x)

        #Non_rigid_registration_layer
        V=torch.where(D>=self.threshold,torch.ones(5,1,32,32),torch.zeros(5,1,32,32))
        U=T*V
        if (turned):
            F = turning(U,anchors)
        else:
            F=U
        R = self.RNN(F)

        return D,R

def turning(U,anchors,treshold = 0.1):
    U_temp = np.array(U)
    height,width,depth = U_temp.shape
    F_temp = np.zeros((32,32,depth))
    for i in range(depth):
        F_temp[:,:,i:i+1]=rotate(U_temp[:,:,i:i+1],anchors[:,:,i])
    F = torch.from_numpy(F_temp)
    return F

def gen_offsets(kernel_size):
    offsets = np.zeros((2, kernel_size * kernel_size), dtype=np.int)
    ind = 0
    delta = (kernel_size - 1) // 2
    for i in range(kernel_size):
        y = i - delta
        for j in range(kernel_size):
            x = j - delta
            offsets[0, ind] = x
            offsets[1, ind] = y
            ind += 1
    return offsets

def rotate(img_base,anchor,kernel_size=3):
    img = resize_120(img_base)
    delta = (kernel_size - 1) // 2
    height,width,depth = img_base.shape
    res = np.zeros((64 * kernel_size, 64 * kernel_size, depth), dtype=np.uint8)
    offsets = gen_offsets(kernel_size)
    for i in range(kernel_size*kernel_size):
        ox, oy = offsets[:, i]
        index0 = anchor[0] + ox
        index1 = anchor[1] + oy
        temp = img[index1, index0,:].reshape(64, 64,depth).transpose(1, 0, 2)
        res[oy + delta::kernel_size, ox + delta::kernel_size,:] = temp
    return resize_32(res)

def resize_32(img):
    height, width, depth = img.shape
    dx,dy = int(height/32),int(width/32)
    res=np.zeros((32,32,depth),dtype=float)
    for x in range(32):
        realx = x*dx
        for y in range(32):
            realy = y*dy
            res[x,y,:]=np.mean(np.mean(img[realx:realx+dx,realy:realy+dy,:],axis=0),axis=0)
    return res
  
def resize_120(img):
    height, width, depth = img.shape
    res=np.zeros((120,120,depth),dtype=float)
    for x in range(120):
        realx = int(x*height/120)
        for y in range(120):
            realy=int(y*width/120)
            res[x,y,:]=img[realx,realy,:]
    return res