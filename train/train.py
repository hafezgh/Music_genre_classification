from utils.miscellaneous import manual_seed
from model import CRNN
import torch.nn as nn
import torch
import numpy as np
import gzip
import pickle
import time
import torch.optim as optim
if (torch.cuda.is_available()):
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def train(hparams, train_loader, val_loader, train_len, val_len, **kwargs):
    manual_seed(2045)
    model = CRNN(len(classes), hparams['c'], hparams['h'], hparams['w'], hparams['k'], hparams['filters'],\
                    hparams['poolings'], hparams['dropout_rate'], gru_units=hparams['gru_units'])
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'])
    try:
        path = kwargs['path']
        stream = gzip.open(path, "rb")
        checkpoint = pickle.load(stream)
        stream.close()
        train_loss = checkpoint['train_loss']
        train_accuracy = checkpoint['train_accuracy']
        val_loss = checkpoint['val_loss']
        val_accuracy = checkpoint['val_accuracy']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_load = checkpoint['epoch']
        print(f'Checkpoint found! Training will resume from epoch {epoch_load+1}')
        print('Last epoch results: ')
        print(f"Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_accuracy[-1]:.2f}")
        print(f'Val Loss: {val_loss[-1]:.4f}, Val Acc: {val_accuracy[-1]:.2f}')
        if 'lr_scheduler' in kwargs.keys() and 'scheduler_state_dict' in checkpoint.keys():
            print('Learning rate sceduler is active.\n')
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1, verbose=True)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    except:
        print('No checkpoints found! Training will start from the beginning.\n')
        train_loss, train_accuracy = [], []
        val_loss, val_accuracy = [], []
        epoch_load = 0
        scheduler = None
        es = False
        if 'lr_scheduler' in kwargs.keys():
            print('Learning rate sceduler is active.\n')
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1, verbose=True)

    es = False
    if 'early_stopping' in kwargs.keys():
        print('Early stopping is active.')
        print()
        es = True
        min_val_loss = np.inf
        patience = 30
        epochs_no_improve = 0
        best_model = None

    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(hparams['epochs']-epoch_load):
        print(f"Epoch {epoch+epoch_load+1} of {hparams['epochs']}")
        train_epoch_loss, train_epoch_accuracy = fit(
            model, train_loader, train_len, optimizer, criterion
        )
        val_epoch_loss, val_epoch_accuracy = validate(
            model, val_loader, val_len, criterion
        )
        if scheduler:
            scheduler.step()
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

        if es:
            if val_epoch_loss < min_val_loss:
                #Saving the model
                min_val_loss = val_epoch_loss
                best_model = copy.deepcopy(model.state_dict())
                #print('Min loss %0.2f' % min_loss)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == patience:
                print(f'Early stopping after {epoch+epoch_load+1} epochs!')
                model.load_state_dict(best_model)
                break
                
        print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')
        checkpoint_to_save = {'model_state_dict': model.state_dict(),                                                 
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch+epoch_load,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy
                }
        if scheduler:
            checkpoint_to_save['scheduler_state_dict'] = scheduler.state_dict()
        stream = gzip.open('/content/drive/MyDrive/music_classification/checkpoint_crop2.pt', "wb")
        pickle.dump(checkpoint_to_save, stream)
        stream.close()
    end = time.time()

    print(f"Training time: {(end-start)/60:.3f} minutes")
    return model, train_loss, train_accuracy, val_loss, val_accuracy