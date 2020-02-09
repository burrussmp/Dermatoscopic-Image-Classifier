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
from utils import *
from train import model_generator
def create_cm(predictions, Y, labels='01234567', title='Confusion Matrix'):
    plt.figure()
    m1 = confusion_matrix(Y,predictions, labels=np.array([int(i) for i in labels]))
    m1 = m1.astype('float') / m1.sum(axis=1)[:, np.newaxis]
    m1 = np.round(m1, 2)
    df_cm = pd.DataFrame(m1, index=[i for i in labels],
                         columns=[i for i in labels])
    sn.heatmap(df_cm, annot=True)
    plt.title(title)
    title = title.replace(' ', '_')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.savefig('./cm.png')

def compute_accuracy(y_pred, y_true):
    print('Accuracy', accuracy_score(y_true, y_pred, normalize=True))

def compute_precision(y_pred, y_true):
    print('Precision', precision_score(y_true, y_pred,average='weighted'))

def compute_recall(y_pred, y_true):
    print('Recall', recall_score(y_true, y_pred,average='weighted'))

def test(model, device, test_loader, classWeight_tensor=None):
    model.eval()  # inference mode
    results = np.zeros((len(test_loader),1))
    results_with_rejection = np.zeros((len(test_loader),1))
    i = 0
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:  # load the data
            data, target = data.to(device), target.to(device)
            output = model(data)  # collect the outputs
            prediction,predictions_with_reject = output_transformation(output)
            result = np.argmax(prediction.cpu().data.numpy()[0])
            pred = prediction.argmax(dim=1, keepdim=True)
            results[i] = result
            results_with_rejection[i] = np.argmax(predictions_with_reject.cpu().data.numpy()[0])
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            if i > 0 and i % 100 == 0:
                print("[INFO] predicted {} total images".format(i))
            i += 1
        print(correct / len(test_loader))
    return results,results_with_rejection

def createPlots(model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_folder = os.path.join(DATA, 'test')
    path_train_csv = os.path.join(DATA, 'labels', 'Test_labels.csv')
    x_test, y_test = load_data(train_folder, path_train_csv)
    x_test = preprocess(x_test)
    x_test = reshapeInput(x_test)
    y_test = np.argmax(y_test, axis=1)
    test_dataset = (torch.FloatTensor(x_test), torch.FloatTensor(y_test))
    test_loader = torch.utils.data.DataLoader(TensorDataset(*test_dataset), **kwargs)
    print('testing..')
    y_pred,y_pred_with_rejection = test(model, device, test_loader)
    y_pred = np.squeeze(y_pred)
    y_pred_with_rejection = np.squeeze(y_pred_with_rejection)

    create_cm((y_pred+1)%8,(y_test+1)%8)
    create_cm((y_pred_with_rejection+1)%8,(y_test+1)%8) # rotate to make rejection class 0
    compute_accuracy(y_pred,y_test)
    compute_precision(y_pred,y_test)
    compute_recall(y_pred,y_test)

    compute_accuracy(y_pred_with_rejection,y_test)
    compute_precision(y_pred_with_rejection,y_test)
    compute_recall(y_pred_with_rejection,y_test)
    # plot the training and validation loss
    val_loss = np.load('./val_loss.npy')
    train_loss = np.load('./train_loss.npy')
    epochs = np.arange(len(val_loss))
    plt.figure()
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs,val_loss,label='Validation Loss')
    plt.plot(epochs,train_loss,label='Training Loss')
    plt.legend()
    plt.savefig('./training_curve.png')
    plt.show()


def main():
    # parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    pathToModel = './weights.pt'
    parser = argparse.ArgumentParser(description='Assignment 1')
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    # attempt to use GPU if available
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123456789)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # attempt to use GPU if available
    model = model_generator()
    model.eval()
    if (use_cuda):
        model.cuda()
    if os.path.isfile(pathToModel):
        print('Loading')
        if not use_cuda:
            model.load_state_dict(torch.load(pathToModel, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(pathToModel))

    if (len(args.files) == 0):
        createPlots(model)
    else:
        for file in args.files:
            assert os.path.isfile(file),\
                print('SKIPPING: File does not exist: ', file)

            img = cv2.imread(file).astype(np.float32)
            img = cv2.resize(img, (90, 120))
            img = np.expand_dims(preprocess(img), axis=0)
            img = reshapeInput(img)
            prediction = model(torch.tensor(img).to(device))
            if (use_cuda):
                result = prediction.cpu().data.numpy()
            else:
                result = prediction.data.numpy()
            print(np.argmax(result))


if __name__ == '__main__':
    main()
