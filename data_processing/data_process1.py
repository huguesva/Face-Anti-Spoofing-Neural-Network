import matplotlib.pyplot as plt
import matplotlib
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch.backends.cudnn as cudnn
import os
from PIL import Image

def resize_center_eyes(frame,lx,ly,rx,ry):
  img = np.array(frame).transpose(1,0,2)
  res = np.zeros((256,256,3),dtype=float)
  center = (0.5*(lx+rx),0.5*(ly+ry))
  for x in range(256):
    realx = int((x-128)*3 + center[0])
    for y in range(256):
      realy = int((y-100)*3 + center[1])
      res[x,y,:]=np.mean(np.mean(img[realx-1:realx+2,realy-1:realy+2,::-1],axis=0),axis=0)
  res=res.transpose(1,0,2)
  img_png=res
  return res/255,Image.fromarray(img_png.astype('uint8'))

if __name__ == '__main__':
    
    folder = {}
    images={}
    label={}
    indice = 0

    #Processing train files
    for phone in range(1,6):
        for session in range(1,4):
            for user in range(1,21):
                for file in range(1,6):
                    nom= str(user)
                    if user<10:
                        nom = '0'+nom
                    cap = cv2.VideoCapture('/Volumes/G-DRIVE mobile USB/Train_files_OULU/Train_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'.avi')

                    lines = []
                    with open('/Volumes/G-DRIVE mobile USB/Train_files_OULU/Train_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'.txt', 'r') as f:
                        for i in range(5):
                            lines.append(f.readline())

                    currentFrame=0
                    while(currentFrame<5):

                        ret, frame = cap.read()
            
                        list = lines[currentFrame].split(',')
                        lx = int(list[1])
                        ly = int(list[2])
                        rx = int(list[3])
                        ry = int(list[4])

                        images[str(indice)],pil_img=resize_center_eyes(frame, lx, ly, rx, ry)

                        name = '/Volumes/G-DRIVE mobile USB/Train_files_OULU/Train_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'_'+str(currentFrame)+'.png'
                        folder[str(indice)]=name
                        if file==1:
                            label[str(indice)]=1
                        else:
                            label[str(indice)]=0

                        indice += 1

                        currentFrame += 1

                    cap.release()
                    cv2.destroyAllWindows()

    #Processing dev files
    for phone in range(1,6):
        for session in range(1,4):
            for user in range(21,36):
                for file in range(1,6):
                    nom= str(user)
                    if user<10:
                        nom = '0'+nom
                    cap = cv2.VideoCapture('/Volumes/G-DRIVE mobile USB/dev_files/Dev_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'.avi')

                    lines = []
                    with open('/Volumes/G-DRIVE mobile USB/dev_files/Dev_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'.txt', 'r') as f:
                        for i in range(5):
                            lines.append(f.readline())

                    currentFrame=0
                    while(currentFrame<5):

                        ret, frame = cap.read()
            
                        list = lines[currentFrame].split(',')
                        lx = int(list[1])
                        ly = int(list[2])
                        rx = int(list[3])
                        ry = int(list[4])

                        images[str(indice)],pil_img=resize_center_eyes(frame, lx, ly, rx, ry)

                        name = '/Volumes/G-DRIVE mobile USB/dev_files/Dev_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'_'+str(currentFrame)+'.png'
                        folder[str(indice)]=name
                        if file==1:
                            label[str(indice)]=1
                        else:
                            label[str(indice)]=0

                        indice += 1

                        currentFrame += 1

                    cap.release()
                    cv2.destroyAllWindows()

    #Processing test files
    for phone in range(1,6):
        for session in range(1,4):
            for user in range(36,56):
                for file in range(1,6):
                    nom= str(user)
                    if user<10:
                        nom = '0'+nom
                    cap = cv2.VideoCapture('/Volumes/G-DRIVE mobile USB/Test_files_OULU/Test_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'.avi')

                    lines = []
                    with open('/Volumes/G-DRIVE mobile USB/Test_files_OULU/Test_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'.txt', 'r') as f:
                        for i in range(5):
                            lines.append(f.readline())

                    currentFrame=0
                    while(currentFrame<5):

                        ret, frame = cap.read()
            
                        list = lines[currentFrame].split(',')
                        lx = int(list[1])
                        ly = int(list[2])
                        rx = int(list[3])
                        ry = int(list[4])

                        images[str(indice)],pil_img=resize_center_eyes(frame, lx, ly, rx, ry)
        
                        name = '/Volumes/G-DRIVE mobile USB/Test_files_OULU/Test_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'_'+str(currentFrame)+'.png'
                        folder[str(indice)]=name
                        if file==1:
                            label[str(indice)]=1
                        else:
                            label[str(indice)]=0

                        indice += 1

                        currentFrame += 1

                    cap.release()
                    cv2.destroyAllWindows()

    np.savez("images.npz",**images)
    np.savez("label.npz",**label)
