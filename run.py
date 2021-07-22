import torchvision
import torch
import numpy as np
from torchvision import models, datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from model import *
import os
from train import *

### Hyperparameters
inp_shape = (288, 432)
c=3                 # Channel
h = inp_shape[0]    # Frequency
w = inp_shape[1]    # Time
k=3
filters = [64, 128, 256, 512, 512]
poolings = [(2, 2), (2, 2), (2, 2), (4, 1), (4, 1)]
dropout_rate = 0.1      # 0.2
gru_dropout_rate = 0.3
batch_size = 16
gru_units=32
lr = 0.0001
n_classes = 10
patience = 10
epochs = 1000
hparams = {'batch_size': batch_size, 'c': c, 'h':h, 'w':w, 'k':k, 'filters': filters,\
               'poolings': poolings, 'dropout_rate':dropout_rate,\
            'gru_units': gru_units, 'lr': lr, 'dropout_gru':gru_dropout_rate, 'epochs':epochs, 'patience':patience}
data_dir = './drive/MyDrive/music_classification/Data'


trans = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder(data_dir+'/images_one_sec', transform=trans)

train_size = int(np.ceil(len(dataset)*0.8))
val_size = int(np.ceil(len(dataset)*0.1))
test_size = len(dataset) - train_size - val_size
print('train set size:',train_size,'val set size:', val_size, 'test set size:', test_size)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths = [train_size, val_size, test_size])


num_workers = 2
pin_memory = True
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=hparams['batch_size'], num_workers=num_workers, pin_memory=pin_memory)
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=hparams['batch_size'], num_workers=num_workers, pin_memory=pin_memory)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=hparams['batch_size'], num_workers=num_workers, pin_memory=pin_memory)

classes = os.listdir(data_dir+'/images_original')
DEVICE = 'cuda'
model = CRNN_Base(len(classes), hparams['c'], hparams['h'], hparams['w'], hparams['k'], hparams['filters'],\
                hparams['poolings'], hparams['dropout_rate'], gru_units=hparams['gru_units'])
model.to(DEVICE)
model, train_loss, train_accuracy, val_loss, val_accuracy = train(hparams, train_loader, val_loader, train_size, val_size,\
                                                                  lr_scheduler=False,\
                                                                  early_stopping=True,\
                                                                  )#path=checkpointPath)
