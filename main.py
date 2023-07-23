import os
import torch
import numpy as np
from loader.digit_five_loader import *
from ResNet import *
from VGG import *
from digit_five_trainer import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import datasets, models, transforms
from optparse import OptionParser

def get_args():

    parser = OptionParser()
    
    parser.add_option('--batch_size', dest='batch_size', default=128, type='int', help='batch size')
    parser.add_option('--learning_rate', dest='learning_rate', default=0.001, type='float', help='learning rate')
    parser.add_option('--epoch', dest='epoch', default=20, type='int', help='number of maximum training epoch')
    parser.add_option('--dataset', dest='dataset', default="DigitFive", type='str', help='speicify which dataset used for training')
    parser.add_option('--batch_display', dest='batch display', default=50, type='int', help='batch display')
    parser.add_option('--save_freq', dest='model save frequency', default=5, type='int', help='model save frequency')
    parser.add_option('--model_save_dir', dest='checkpoints_save_dir', default='model', type='str', help='the directory to save the training model')

    (options, args) = parser.parse_args()
    return options

if __name__ == "__main__":
    set_seed()
    

    root_path = {'DigitFive':"/mnt/c/Users/wuyr/Downloads/digit_five/", 'TinyImageNet':"/home_new/dataset/tiny-imagenet-200/", 'Cifar100':"/home/viki/Codes/MultiSource/2/multi-source/data_set_2/"}

    digit_five_train = DigitFiveTrain()
    digit_five_train.train_all()
