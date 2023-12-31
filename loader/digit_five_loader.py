import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
# import tensorflow as tf
import os
from scipy import io as sio

class DigitFive:
    def __init__(self, root_path="/mnt/c/Users/wuyr/Downloads/digit_five/", dataname = ['mnist', 'mnistm', 'usps', 'svhn', 'syn'], train_mode = "train"):
        """
        Initialize some variables
        Load labels & names
        define transform
        """
        self.path = root_path
        self.datanames = dataname
        self.train_mode = train_mode
        self.transform = {
            'train': transforms.Compose([                
                transforms.Resize(28),
                transforms.RandomResizedCrop(28),
                transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
            ]),
            'val': transforms.Compose([
                transforms.Resize(28),
                transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
            ]),
        }
        self.toRGB = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Lambda(lambda x: torch.stack([x, x, x], -1))
            ])

        self.data = {name:self.data_process(self.get_data(name)) for name in self.datanames}
        
        

    def data_process(self, data):
        image = torch.from_numpy(data[0].transpose(0,3,1,2)).float()
        transformed_img = self.transform[self.train_mode](image)
        label = torch.from_numpy(data[1]).long() 
        dataset = TensorDataset(transformed_img, label)
        # trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataset

    def get_data(self, dataname):
        '''
        typename = 'mnist'/'mnistm'/'usps'/'svhn'/'syn'
        '''
        match dataname:
            case 'mnist':
                mat = sio.loadmat(f'{self.path}mnist_data.mat')
                data = (np.array((self.toRGB(mat['train_28'].squeeze()))).astype('float32')).transpose(1,2,0,3)
                target = np.argmax((mat['label_train']), axis = 1)
                opt_data = [data, target]

            case 'mnistm':
                mat = sio.loadmat(f'{self.path}mnistm_with_label.mat')
                data = mat['train']
                target = np.argmax((mat['label_train']), axis = 1)
                opt_data =[data, target]

            case 'usps':
                mat = sio.loadmat(f'{self.path}usps_28x28.mat')
                data = (np.array((self.toRGB(mat['dataset'][0][0].squeeze()))).astype('float32')).transpose(1,2,0,3)
                target = mat['dataset'][0][1].flatten()
                opt_data = [data, target]

            case 'svhn':
                mat = sio.loadmat(f'{self.path}svhn_train_32x32.mat')
                data = mat['X'].transpose(3,0,1,2)
                target = (mat['y']-1).flatten()
                opt_data = [data, target]

            case 'syn':
                mat = sio.loadmat(f'{self.path}syn_number.mat')
                data = mat['train_data']
                target = mat['train_label'].flatten()
                opt_data = [data, target]

        return opt_data

    # def __len__(self):
    #     """
    #     Get the length of the entire dataset
    #     """
    #     print("Length of dataset is ", self.image_labels.shape[0])
    #     return self.image_labels.shape[0]

    # def __getitem__(self, idx):
    #     """
    #     Get the image item by index
    #     """
    #     image, label = self.data[idx]
    #     transformed_img = self.transform[self.train_mode](image)
    #     sample = {'image':transformed_img, 'label':label}
    #     return sample
