import torch
import torch.nn as nn
import torch.nn.functional as F

## Model Settings
RANDOM_SEED = 1
LEARNING_RATE = lrate
BATCH_SIZE = hyper_parameters[7]
NUM_EPOCHS = 15  
NUM_FEATURES = 32*32    
NUM_CLASSES = 22  # number of classes in the output

# hyperparameters
filter_num1 = 64
filter_num2 = 128
filter_num3 = 256
filter_num4 = 512
hidden_layer1 = 4096
dim = 32 # need to be larger, paper start from 224


class VGG(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(VGG, self).__init__() # inheritance??
        
        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1 
        else:
            in_channels = 3  # RGB  3 channels

        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels, filter_num1 , kernel_size= 3,padding= 1), # why padding?
            nn.Tanh(),
            nn.Conv2d(filter_num1, filter_num1 , kernel_size= 3,padding= 1), 
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(filter_num1, filter_num2 , kernel_size= 3,padding= 1), 
            nn.Tanh(),
            nn.Conv2d(filter_num2, filter_num2 , kernel_size= 3,padding= 1), 
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(filter_num2, filter_num3, kernel_size= 3,padding= 1), 
            nn.Tanh(),
            nn.Conv2d(filter_num3, filter_num3, kernel_size= 3,padding= 1), 
            nn.Tanh(),
            nn.Conv2d(filter_num3, filter_num3, kernel_size= 3,padding= 1), 
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(filter_num3, filter_num4, kernel_size=hyper_parameters[1],padding=int((hyper_parameters[1]-1)/2)), # why padding?
            nn.Tanh(),
            nn.Conv2d(filter_num4, filter_num4, kernel_size=hyper_parameters[1],padding=int((hyper_parameters[1]-1)/2)), # why padding?
            nn.Tanh(),
            nn.Conv2d(filter_num4, filter_num4, kernel_size=hyper_parameters[1],padding=int((hyper_parameters[1]-1)/2)), # why padding?
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(filter_num3, filter_num4, kernel_size=hyper_parameters[1],padding=int((hyper_parameters[1]-1)/2)), # why padding?
            nn.Tanh(),
            nn.Conv2d(filter_num4, filter_num4, kernel_size=hyper_parameters[1],padding=int((hyper_parameters[1]-1)/2)), # why padding?
            nn.Tanh(),
            nn.Conv2d(filter_num4, filter_num4, kernel_size=hyper_parameters[1],padding=int((hyper_parameters[1]-1)/2)), # why padding?
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

        )
        self.classifier = nn.Sequential(
            nn.Linear(int(filter_num4 *(dim/32)*(dim/32)),hidden_layer1),
            nn.Tanh(),
            nn.Linear(hidden_layer1, hidden_layer1),
            nn.Tanh(),
            nn.Linear(hidden_layer1, num_classes),
            nn.Tanh(),
            nn.Linear(num_classes, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas