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
#from utils import preprocess,load_dataset
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


def output_transformation(predictions):
    lam = torch.tensor(50.0)
    Ok = torch.exp(-1*predictions)
    top = Ok*(1+torch.exp(lam)*Ok)
    bottom = torch.prod(1+torch.exp(lam)*Ok, dim=1)
    reject = 1.0/bottom
    predictions = torch.div(top.t(), bottom).t()
    predictions_with_reject = torch.cat((predictions, reject.unsqueeze(1)), 1)
    return predictions, predictions_with_reject


def load_data(basePath, csvPath):
    data = []
    labels = []
    rows = open(csvPath).read().strip().split("\n")[1:]
    # random.shuffle(rows)
    for (i, row) in enumerate(rows):
        # check to see if we should show a status update
        if i > 0 and i % 100 == 0:
            print("[INFO] processed {} total images".format(i))
        row_arr = row.strip().split(",")
        imagePath = row_arr[0] + '.jpg'
        label = row_arr[1::]
        imagePath = os.path.join(basePath, imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (256, 256))
        data.append(image)
        labels.append(label)
    data = np.array(data)

    labels = np.array(labels).astype(np.uint8)
    return (data, labels)


def preprocess(X):
    X = X.astype(np.float32)
    return X/255.


def reshapeInput(X):
    return np.swapaxes(X, 1, 3)


def Metrics_Soft_Loss(y_pred, y_true, weights=None):
    lam = torch.tensor(50)
    indices = y_true.type(dtype=torch.int64)
    y_pred = y_pred.type(dtype=torch.float32)
    y_true = y_true.type(dtype=torch.float32)
    d = y_pred.gather(1, indices.type(dtype=torch.int64).view(-1, 1))
    d = torch.squeeze(d)
    if (type(weights) != type(None)):
        d *= weights[indices]
    y_pred = torch.log(1 + torch.exp(lam - y_pred))
    if (type(weights) != type(None)):
        S = torch.squeeze(torch.mm(y_pred, torch.unsqueeze(torch.tensor(
            1) - weights, 1).type(dtype=torch.float32))) - torch.log(1+torch.exp(lam-d))
    else:
        S = torch.sum(y_pred, dim=1) - torch.log(1+torch.exp(lam-d))
    S = torch.squeeze(S)
    y = torch.sum(d + S)
    return y


class RBF(nn.Module):
    def __init__(self, in_features, out_features):
        super(RBF, self).__init__()
        self.num_centers = out_features
        centers = torch.rand(in_features, out_features)*2-1
        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(1, out_features)/10)

    def kernel_fun(self, batches):
        diff = batches.unsqueeze(2)-self.centers
        l2 = torch.sum(torch.pow(diff, 2), dim=1)
        return l2

    def forward(self, batches):
        return self.kernel_fun(batches)


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
