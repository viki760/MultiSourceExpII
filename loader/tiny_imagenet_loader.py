from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image


import torch
import torch.nn as nn
import  torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import os



class TinyImageNet(Dataset):
    def __init__(self, type='train', class_list = range(200)):
        '''
        input: type = train/val/test
               class_list = list of classes for this task (should be a subset of range(200))
        output: torch.Dataset
        '''
        self.samples_path = []
        self.label=[]
        path=f"/home_new/dataset/tiny-imagenet-200/{type}"
        for imagenet_class in class_list:
            imagenet_class = str(imagenet_class)
            if os.path.isdir(os.path.join(path, imagenet_class)):
                for image_file in os.listdir(os.path.join(path, imagenet_class)):
                    if os.path.isdir(os.path.join(path, imagenet_class, image_file)):
                        for image in os.listdir(os.path.join(path, imagenet_class, image_file)):
                            if image.endswith(".JPEG"):
                                self.samples_path.append(os.path.join(path, imagenet_class, image_file, image))
                                self.label.append(int(imagenet_class))
        
        self.transform = transforms.Compose([
                        transforms.ToTensor(),             
                        transforms.RandomResizedCrop(64) if type == 'train' else transforms.Lambda(lambda x: x) ,
                        transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
                    ])    

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        image_data_path=self.samples_path[idx]
        # image dim 3*64*64
        image = Image.open(image_data_path).convert('RGB')
        image = self.transform(image)
        label=self.label[idx]

        return image, label
    
if __name__ == '__main__':
    data = TinyImageNet()
    data[0]
    type(data)
