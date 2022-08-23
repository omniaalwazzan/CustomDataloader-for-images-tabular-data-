import torch
from torchvision import transforms
import pandas as pd
import os 
import PIL 
from PIL import Image 


trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])


device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self, csv_path, images_folder, transform=None):
        df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.images_name = df['file_name']
        
        self.y = df['label']
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        img_path=os.path.join(self.images_folder,
                              self.images_name[index])
        image= Image.open(img_path).convert('RGB')
               
        y_label = self.y[index]
      
        
        if self.transform is not None:
            image = self.transform(image)
        return (image, y_label)



def load_data(csv_file, image_dir):
    train_data =CustomDataset(csv_file, image_dir,transform=trans)
    
    return train_data


