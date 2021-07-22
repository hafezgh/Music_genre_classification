import torch
DEVICE = 'cuda'
import math
import torch.optim as optim
from model import *
import os
import copy, gzip, pickle, time
data_dir = './drive/MyDrive/music_classification/Data'
classes = os.listdir(data_dir+'/images_original')


def fit(model, train_loader, train_len, optimizer, criterion):
    model.train()
    batch_size = train_loader.batch_size
    n_batches = math.ceil(train_len/batch_size)
    #print('Batch Size:', batch_size,'Number of Batches:', n_batches)
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    #prog_bar = tqdm(enumerate(train_loader), total=int(train_len/batch_size))
    for i, data in enumerate(train_loader):
        counter += 1
        data, target = data[0].to(DEVICE), data[1].to(DEVICE)
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss / counter
    train_accuracy = 100. * train_running_correct / total
    return train_loss, train_accuracy

def validate(model, val_loader, val_len, criterion):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    batch_size = val_len
    #prog_bar = tqdm(enumerate(val_loader), total=int(val_len/batch_size))
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            counter += 1
            data, target = data[0].to(DEVICE), data[1].to(DEVICE)
            total += target.size(0)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss / counter
        val_accuracy = 100. * val_running_correct / total
        return val_loss, val_accuracy

def train(hparams, train_loader, val_loader, train_len, val_len, checkpoint_path=None, **kwargs):
    model = CRNN_Base(len(classes), hparams['c'], hparams['h'], hparams['w'], hparams['k'], hparams['filters'],\
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
            if kwargs['lr_scheduler'] == True:
                print('Learning rate sceduler is active.\n')
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1, verbose=True)
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                scheduler = False
        else:
            scheduler = False
    except:
        print('No checkpoints found! Training will start from the beginning.\n')
        train_loss, train_accuracy = [], []
        val_loss, val_accuracy = [], []
        epoch_load = 0
        scheduler = None
        es = False
        if 'lr_scheduler' in kwargs.keys():
            if kwargs['lr_scheduler'] == True:
                print('Learning rate sceduler is active.\n')
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1, verbose=True)
            else:
                scheduler = False
        else:
            scheduler = False

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
        ## Saving the model
        if checkpoint_path != None:
            stream = gzip.open(checkpoint_path, "wb")
            pickle.dump(checkpoint_to_save, stream)
            stream.close()
    end = time.time()

    print(f"Training time: {(end-start)/60:.3f} minutes")
    return model, train_loss, train_accuracy, val_loss, val_accuracy