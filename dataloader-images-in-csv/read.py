import pandas as pd
from torchvision import models
import torch.optim as optim
from torch import nn
from data_generator import load_data
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import torch
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchvision import transforms

import numpy as np



# for rocks sisscor and paper exp
# load_data recives the csv file containing images name, and the images directory
tr_data = load_data(r'.\rsp_all.csv',r'.\rps')
v_data = load_data(r'.\rsp_all_valid.csv',r'.\rps_val')
#print('images in train data are', tr_data.images_name[0])
#print(len(tr_data))


train_loader = DataLoader(dataset=tr_data,batch_size=20,shuffle=True)
val_loader = DataLoader(dataset=v_data,batch_size=20,shuffle=False)

# to test the data loader if it works properly or not
b=iter(train_loader)
image, lable=b.next()


labels_map = {
    0: "Rock",
    1: "Scissors",
    2: "Paper",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

# this for loop will plot the image with its corisponing label if the loader was working proberly
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(tr_data), size=(1,)).item()
    img, label = tr_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))


