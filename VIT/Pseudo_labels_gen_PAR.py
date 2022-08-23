#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import cv2
import PAR
import warnings
import numpy as np
import pandas as pd
from math import ceil
from time import time
from skimage import color
from tqdm.auto import tqdm
import multiprocessing as mp
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

import timm
import torch
import torchvision
import torchsummary
from torch import nn
from torch.optim import Adam
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad

#get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings("ignore")

# In[3]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device


# In[4]:


training_images = '/home/zephyr/Desktop/Newcastle_University/11_FP_D/Dataset/Dataset/1.training/1.training'
training_pseudo_labels = '/home/zephyr/Desktop/Newcastle_University/11_FP_D/Dataset/training_pseudo_labels'


# In[5]:


def preprocess_image(img: np.ndarray, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], resized = (224, 224)) -> torch.Tensor:
  
  preprocessing = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Resize((224, 224)),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                      ])
  return preprocessing(img.copy()).unsqueeze(0)


# In[6]:


def reshape_transform(tensor, height=7, width=7):
  result = tensor[:, 1 :  , :].reshape(tensor.size(0),
                                       height, width, tensor.size(2))
  
  # Bring the channels to the first dimension,
  # like in CNNs.
  result = result.transpose(2, 3).transpose(1, 2)
  return result


# In[7]:


def mask_threshold(mask):
    v = np.concatenate(mask)
    t = v.mean()
    d = np.inf
    ds = 0.005
    while d > ds:
        g1 = v[v>t]
        g2 = v[v<=t]
        m1 = g1.mean()
        m2 = g2.mean()
        tp = (m1 + m2)/2
        d = np.abs(t - tp)
        t = tp
        #print(t)

    imt = mask > t

    return imt


# In[8]:


model_vit_base_patch16_224 = timm.create_model('vit_base_patch32_224', pretrained = False, num_classes = 3,
                                               drop_rate = 0.2, attn_drop_rate = 0.2).to(device)

model_vit_base_patch16_224.load_state_dict(torch.load(f='/home/zephyr/Desktop/Newcastle_University/11_FP_D/Models/model_vit_base_patch32_224_2.pth', map_location=device))


# In[9]:


model_vit_base_patch16_224.eval()


# In[10]:


target_layer = [model_vit_base_patch16_224.blocks[-1].norm1]
#target_layer


# In[11]:


cam = GradCAM(model = model_vit_base_patch16_224, target_layers=target_layer,
              reshape_transform=reshape_transform)


# In[12]:


par = PAR.PAR(num_iter=15, dilations=[1,2,4,8,16,32])
par.to(device)


# In[25]:


def cam_refine(cam, image, image_tensor):
    cam = par(image_tensor, transforms.ToTensor()(cam).unsqueeze(0))
    cam = minmax_scale(cam.squeeze(0).squeeze(0).cpu().detach().numpy().reshape(-1, 1))
    cam = cam.reshape(224, 224)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    
    cam = cam*thresh
    cam = mask_threshold(cam).astype(np.float32)
    
    return cam


# In[26]:


# function that loads images from a directory generates cam for them and saves them to a directory
def generate_pseudo_labels(images_dir, pseudo_labels_dir, model, CAM):
    for image_name in tqdm(os.listdir(images_dir), bar_format='{l_bar}{bar:100}{r_bar}{bar:-100b}'):
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)
        size = image.shape[:-1]
        image = cv2.resize(image, (224, 224))
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.to(device)
        #print(image_name)
        label = list(map(int, image_name[-13:-4].strip('][').split(', ')))
        
        pseudo_label = np.zeros((224, 224, 4))
        
        #print(image_name)
        
        if label[0] == 1:
            cam_0 = CAM(input_tensor=image_tensor, targets=[ClassifierOutputTarget(np.array([0]))])[0, :]
            pseudo_label[:, :, 0] = cam_refine(cam_0, image, image_tensor)
        
        if label[1] == 1:
            cam_1 = CAM(input_tensor=image_tensor, targets=[ClassifierOutputTarget(np.array([1]))])[0, :]
            pseudo_label[:, :, 1] = cam_refine(cam_1, image, image_tensor)
        
        if label[2] == 1:
            cam_2 = CAM(input_tensor=image_tensor, targets=[ClassifierOutputTarget(np.array([2]))])[0, :]
            pseudo_label[:, :, 2] = cam_refine(cam_2, image, image_tensor)
        
        pseudo_label[:, :, 3] = 1 - pseudo_label.max(axis=-1)
        
        pseudo_label_path = os.path.join(pseudo_labels_dir, image_name)
        pseudo_label = pseudo_label
        
        pseudo_label = cv2.resize(pseudo_label.astype(np.float32), (size[1], size[0]))
        
        cv2.imwrite(os.path.join(pseudo_labels_dir, image_name), pseudo_label*255)    # on loading the mask from the directory, normalize by dividing by 255 to get binary mask
        #print(f'{image_name} done')


# In[ ]:


generate_pseudo_labels(training_images, training_pseudo_labels, model_vit_base_patch16_224, cam)


# In[ ]:





# In[ ]:




