import os
import torch
import numpy as np
from loader.digit_five_loader import *
from ResNet import *
from VGG import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import datasets, models, transforms

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.Generator().manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


class Train:
    def __init__(self, root_path="/mnt/c/Users/wuyr/Downloads/digit_five/", dataname = ['mnist', 'mnistm', 'usps', 'svhn', 'syn'], model_name = "vgg16", number_classes = 10, path="/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExpII/model/", loadPretrain=0):
        """
        Init Dataset, Model and others
        """
        self.save_path = path
        self.dataname = dataname
        dataset = DigitFive(root_path=root_path, dataname = ['mnist', 'mnistm', 'usps', 'svhn', 'syn'], train_mode = "train")
        self.dataset = dataset.data
        if model_name == "resnet50":
            self.model = resnet50(pretrained=(loadPretrain==1), num_classes = number_classes, model_path = path)
        elif model_name == "vgg16":
            self.model = vgg16(pretrained=(loadPretrain==1), num_classes = number_classes, model_path = path)

        if torch.cuda.device_count() > 1:
            print("There are ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count() == 1:
            print("There is only one GPU")
        else:
            print("Only use CPU")

        if torch.cuda.is_available():
            self.device = 'cuda'
            self.model.cuda()
        else:
            self.device = 'cpu'

    def start_train(self, data_name, epoch=1, batch_size=128, learning_rate=0.001, batch_display=50, save_freq=10):
        """
        Detail of training
        """
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.lr = learning_rate
        
        loss_function = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epoch_num):
            epoch_count = 0
            total_loss = 0
            total_num = len(self.dataset[data_name])
            train_dataset, test_dataset = torch.utils.data.random_split(self.dataset[data_name], [int(0.9 * total_num), int(0.1 * total_num)])
            dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) # num_workers=8
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
            
            for i_batch, sample_batch in enumerate(dataloader):
 
                # Step.1 Load data and label
                images_batch, labels_batch = sample_batch[0], sample_batch[1]
                """
                for i in range(images_batch.shape[0]):
                    img_tmp = transforms.ToPILImage()(images_batch[i]).convert('RGB')
                    plt.imshow(img_tmp)
                    plt.pause(0.001)
                """                   
                labels_batch = torch.LongTensor(labels_batch.view(-1).numpy())
                # if torch.cuda.is_available():
                #     input_image = autograd.Variable(images_batch.cuda())
                #     target_label = autograd.Variable(labels_batch.cuda(non_blocking=True))
                # else:
                #     input_image = autograd.Variable(images_batch)
                #     target_label = autograd.Variable(labels_batch)
                
                input_image, target_label = images_batch.to(self.device), labels_batch.to(self.device)
                
                # Step.2 calculate loss
                self.model.train()
                output = self.model(input_image)
                loss = loss_function(output, target_label)
                epoch_count += 1
                total_loss += loss
                # Step.3 Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Check Result
                if i_batch % batch_display == 0:
                    self.model.eval()
                    correct = 0
                    for test_image, test_label in test_dataloader:
                        test_image, test_label = test_image.to(self.device), test_label.to(self.device)
                        pred_prob, pred_label = torch.max(self.model(test_image), dim=1)
                        # print("Input Label : ", test_label[:10])
                        # print("Output Label : ", pred_label[:10])
                        correct += (pred_label == test_label).sum()
                    
                    batch_acc = correct/len(test_dataset)
                    print("Epoch : %d, Batch : %d, Loss : %f, Batch Accuracy %f" %(epoch, i_batch, loss, batch_acc))
            """
            Save model
            """
            print("Epoch %d Average Loss : %f" %(epoch, total_loss * self.batch_size / epoch_count))
            if epoch % save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, f'{data_name}_model.pkl'))
                           
    def train_all(self, epoch=20, batch_size=128, learning_rate=0.001, batch_display=50, save_freq=5):
        for name in self.dataname:
            print(f"---------------------------------{name}--------------------------------")
            self.start_train(name, epoch, batch_size, learning_rate, batch_display, save_freq)
         
                           
if __name__ == '__main__':
    set_seed()
    train = Train()
    train.train_all()