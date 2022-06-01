import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import PIL
from PIL import Image
import os
import torchvision.transforms as transforms



pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(pretrained_size, padding=2),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=pretrained_means,
                                                std=pretrained_stds)
                       ])




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
        self.class2index = {"grade_2": 0, "grade_3": 1,"grade_4":2} # our dataset has 3 classes

    def __len__(self):
        return len(self.df) # length of the dataset

    def __getitem__(self, index):
        filename = self.df.loc[index]["Slide_ID"] # this is the col name of the img path e.g. 123.png
        label = self.class2index[self.df.loc[index]["label"]] # this is the col name of the label its content is grade_2,grade_3...etc
        image = PIL.Image.open(os.path.join(self.images_folder, filename)) # we need to add path.join or it won't work
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# this function will receive the file contains images path with corresponding labels, and image folder contains all images 
def load_data(csv_file,img_Dir): 
    train_dataset = CustomDataset(csv_file, img_Dir, transform=train_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=2)
    return train_loader

