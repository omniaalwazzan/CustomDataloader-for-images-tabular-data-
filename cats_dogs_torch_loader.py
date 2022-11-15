csv_path = r'../dogs_panda_cats_lable.csv'
image_dir = r'../cat_pan_dog'
batch_size=30
NUM_WORKERS=0
PIN_MEMORY=True

import torch
from torchvision import transforms
import pandas as pd
import os 
import PIL 
import cv2
import numpy as np
from PIL import Image 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



# means = [0.485, 0.456, 0.406]
# stds = [0.229, 0.224, 0.225]
# NUM_WORKERS=0



# transforms = transforms.Compose([
#                     transforms.RandomHorizontalFlip(0.5),
#                     transforms.RandomVerticalFlip(0.5),
#                     transforms.Resize((220,220)),
#                     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



# train_transforms = transforms.Compose([
#                            transforms.Resize((224,224)),
#                            transforms.ToTensor(),
#                            transforms.Normalize(mean=means,
#                                                 std=stds),
#                        ])

transforms =  transforms.Compose([
      transforms.Resize((220,220)),
      transforms.PILToTensor(),
      transforms.ConvertImageDtype(torch.float),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225),)
  ])


device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomDataset1(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
        self.class2index = {0,1,2}

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.loc[index]["FILENAME"]
        label = self.df.loc[index]["gt"]
        image = PIL.Image.open(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label,filename



def load_data(df1, image_dir):

    train_data = CustomDataset1(df1 ,image_dir,transform=transforms)
    
    return train_data

def Data_Loader(df1,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY): 
    test_ids = CustomDataset1(df1 ,image_dir,transform=transforms)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader


def main():
    tr_data =load_data(csv_path, image_dir) # we need just for visualization 
    train_loader= Data_Loader(csv_path,image_dir,batch_size=batch_size) 
    
    a=iter(train_loader)
    # a1=next(a)
    # img1=a1[0][1,0,:,:].numpy()
    # gt=a1[1][1].numpy() # when patch size 2 was chosen, c_value here represents the second 
    # name=a1[2][1]
    
    
    image, lable, img_name=a.next()
    
    
    labels_map = {
        0: "cat",
        1: "dog",
        2: "panda",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(tr_data), size=(1,)).item()
        img, label,img_name = tr_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        
        
if __name__ == "__main__":
    main()
