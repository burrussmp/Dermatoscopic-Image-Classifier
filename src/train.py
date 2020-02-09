from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import cv2
# from utils import preprocess,load_dataset
from torch.utils.data import TensorDataset, Dataset
import math
import torchvision.models as models
from torchvision import transforms
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from matplotlib import pyplot as plt
from skimage import color
from config import *
from utils import *


def model_generator():
    model = models.resnet101()
    model.layer4[2] = nn.Sequential(
        nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Tanh()
    )
    model.fc = nn.Sequential(
        RBF(2048,7)
    )
    return model

def train(model, device, train_loader, optimizer, epoch):
    model.train() # training  mode
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    i = 0
    for batch_idx, (data, target) in enumerate(train_loader): # iterate across training dataset using batch size
        data, target = data.to(device), target.to(device) #
        #print(target.shape)
        #target = target.squeeze(1)
        optimizer.zero_grad() # set gradients to zero
        output = model(data) # get the outputs of the model
        loss = Metrics_Soft_Loss(output,target)
        #loss = criterion(output,target.long())
        total_loss += loss
        loss.backward() # Accumulate the gradient
        optimizer.step() # based on currently stored gradient update model params using optomizer rules
        i += 1
        if batch_idx % 10 == 0: # provide updates on training process
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss.item()/i))
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss
def validate(model, device, validation_loader):
    model.eval() # inference mode
    test_loss = 0
    correct = 0
    #criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in validation_loader: # load the data
            data, target = data.to(device), target.to(device)
            output = model(data) # collect the outputs
            loss = Metrics_Soft_Loss(output,target)
            #loss = criterion(output,target.long())
            test_loss += loss
            pred = output.argmin(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.long().view_as(pred)).sum().item()

    test_loss /= len(validation_loader.dataset) # compute the average loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))
    return test_loss


def main():

    # Training settings
    batch_size = 8
    learning_rate = 0.0001
    gamma = 0.5
    epochs = 50
    lr_scheduler_step_size = 12
    adam_betas = (0.9, 0.999)
    pathToModel = os.path.join(BASEDIR, 'weights.pt')
    restart = True

    # attempt to use GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # CPY ABOVE HERE

    train_folder = os.path.join(DATA, 'train')
    path_train_csv = os.path.join(DATA, 'labels', 'Train_labels.csv')

    print('Loading training data...')
    trainX, trainY = load_data(train_folder, path_train_csv)
    print('x train shape:', trainX.shape)

    print('Split the train/val data sets 80/20')
    num = int(trainX.shape[0]*0.2)
    np.random.seed(1234567)
    idxs = np.random.choice(np.arange(trainX.shape[0]), num, replace=False)

    x_val_raw = trainX[idxs]
    y_val = trainY[idxs]
    x_train_raw = np.delete(trainX, idxs, axis=0)
    y_train = np.delete(trainY, idxs, axis=0)
    y_train = np.argmax(y_train, axis=1)
    y_val = np.argmax(y_val, axis=1)

    x_train = preprocess(x_train_raw)
    x_val = preprocess(x_val_raw)
    print('Reshaping to have channels first')
    x_train = reshapeInput(x_train)
    x_val = reshapeInput(x_val)

    print('Number of training data:', x_train_raw.shape[0])
    print('Number of validation data:', x_val_raw.shape[0])

    num = int(trainX.shape[0]*0.2)
    np.random.seed(1234567)
    idxs = np.random.choice(np.arange(trainX.shape[0]), num, replace=False)
    x_val_raw = trainX[idxs]
    y_val = trainY[idxs]
    x_train_raw = np.delete(trainX, idxs, axis=0)
    y_train = np.delete(trainY, idxs, axis=0)

    y_train = np.argmax(y_train, axis=1)
    y_val = np.argmax(y_val, axis=1)

    # preprocess training and validation
    print('Preprocessing...')
    x_train = preprocess(np.copy(x_train_raw))
    x_val = preprocess(np.copy(x_val_raw))
    print('Reshaping to have channels first')
    x_train = reshapeInput(x_train)
    x_val = reshapeInput(x_val)
    # load the model
    model = model_generator()
    # model = PhoneLocator().to(device)
    if (use_cuda):
        model.cuda()
    # load the optimizer and setup schedule to reduce learning rate every 10 epochs
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, betas=adam_betas)
    scheduler = StepLR(
        optimizer, step_size=lr_scheduler_step_size, gamma=gamma)

    train_dataset = (torch.FloatTensor(x_train),torch.FloatTensor(y_train))
    validation_dataset = (torch.FloatTensor(x_val),torch.FloatTensor(y_val))
    # create train and validatoin data loader
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor()
    ])

    train_dataset = CustomTensorDataset(tensors=train_dataset, transform=data_transform)
    histcount = np.histogram(y_train,bins=7)[0]
    classWeight = 1.0 - histcount / histcount.sum()
    classWeight_tensor = torch.FloatTensor(classWeight).to(device)
    samples_weights = classWeight_tensor[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,sampler=sampler, **kwargs)
    validation_loader = torch.utils.data.DataLoader(TensorDataset(*validation_dataset), shuffle=True,**kwargs)
    # load model if path exists
    if os.path.isfile(pathToModel) and not restart:
        print('restarting..')
        model.load_state_dict(torch.load(pathToModel))
    # each iteration gather the n=test_batch_size samples and their respective labels [0,9]
    best_loss = math.inf
    train_loss_save = np.zeros((epochs))
    val_loss_save = np.zeros((epochs))


    print('Beginning to train')
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        val_loss = validate(model, device, validation_loader)
        if (use_cuda):
            train_loss_save[epoch-1] = train_loss.cpu().data.numpy()
            val_loss_save[epoch-1] = val_loss.cpu().data.numpy()
        else:
            train_loss_save[epoch-1] = train_loss.data.numpy()
            val_loss_save[epoch-1] = val_loss.data.numpy()
        if (val_loss < best_loss):
            print('Loss improved from ', best_loss, 'to',val_loss,': Saving new model to',pathToModel)
            best_loss = val_loss
            torch.save(model.state_dict(), pathToModel)
        scheduler.step()
        np.save('./val_loss.npy',val_loss_save)
        np.save('./train_loss.npy',train_loss_save)


if __name__ == '__main__':
    main()
