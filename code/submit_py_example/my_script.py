import os, time, sys, torch, torchvision, matplotlib
import numpy as np
import pandas as pd 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
from torchvision import datasets 
from torchvision import transforms
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from PIL import Image

print("Argument list:", str(sys.argv))
