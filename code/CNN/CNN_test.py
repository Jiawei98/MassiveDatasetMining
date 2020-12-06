#!/usr/bin/env python

import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

print(1)

# Get Argument
hyper_parameters = pd.read_csv("randomSearchCNN.csv").iloc[7].values.tolist()
lrate = hyper_parameters[6]
hyper_parameters = [int(i) for i in hyper_parameters]
## Model Settings
RANDOM_SEED = 1
LEARNING_RATE = lrate
BATCH_SIZE = hyper_parameters[7]
NUM_EPOCHS = 15
NUM_FEATURES = 32*32
NUM_CLASSES = 22

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
    
GRAYSCALE = True

print(2)

## Load Dataset
resize_transform = transforms.Compose([transforms.Grayscale(),transforms.Resize((32, 32)),transforms.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(root="dataset/images_train", 
                                                 transform=resize_transform)
validation_dataset = torchvision.datasets.ImageFolder(root="dataset/images_validation", 
                                                transform=resize_transform)
train_dataset = torch.utils.data.ConcatDataset([train_dataset, validation_dataset])
test_dataset = torchvision.datasets.ImageFolder(root="dataset/images_test", 
                                                transform=resize_transform)


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)


device = torch.device(DEVICE)
torch.manual_seed(0)
print("End of loading data.")

## Model
class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()
        
        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels, hyper_parameters[0], kernel_size=hyper_parameters[1],padding=int((hyper_parameters[1]-1)/2)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(hyper_parameters[0], hyper_parameters[2], kernel_size=hyper_parameters[3],padding=int((hyper_parameters[3]-1)/2)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(hyper_parameters[2]*8*8),hyper_parameters[4]),
            nn.Tanh(),
            nn.Linear(hyper_parameters[4], hyper_parameters[5]),
            nn.Tanh(),
            nn.Linear(hyper_parameters[5], num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

torch.manual_seed(RANDOM_SEED)

model = LeNet5(NUM_CLASSES, GRAYSCALE)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  

## Model Training
def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
    
train_acc=np.array([])
test_acc=np.array([])

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        optimizer.step()
        
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), cost))

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        acc1=compute_accuracy(model, train_loader, device=DEVICE)
        acc2=compute_accuracy(model, test_loader, device=DEVICE)
        train_acc=np.append(train_acc, acc1)
        test_acc=np.append(test_acc, acc2)
        print('Epoch: %03d/%03d | Train: %.3f%% | Test: %.3f%%' % (
              epoch+1, NUM_EPOCHS, acc1, acc2))

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

## Model Evaluation

print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=DEVICE)))

save = np.array([train_acc[len(train_acc)-1],test_acc[len(test_acc)-1]])

np.savetxt("res_test.csv",save, delimiter=",")

## Plotting
plt.figure(figsize=(7, 5))
plt.plot(train_acc, c='#deb068', label='Training Accuracy')
plt.plot(test_acc, c='#595857', label='Test Accuracy')
plt.legend(loc=4, prop={'weight' : 'normal','size': 12})
plt.title('Accuracy in each Epoch', fontsize=15)
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Accuracy", fontsize=13)
plt.savefig('accuracy.png')
