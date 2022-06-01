from torch import nn
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import torch.optim as optim
from tqdm import tqdm
import timm
import numpy as np

# internal import 
from data_loader import load_data

# This file will just train the model and save it ,val and test loaders are not available  

file1_csv=r"C:\Users\omnia\PycharmProjects\Clam\data\NewExp\flowers\imgs.csv"
file2_image= r"C:\Users\omnia\PycharmProjects\Clam\data\NewExp\flowers\train\images"
train_loader=load_data(file1_csv,file2_image)

print("The length of the Training DataLoader is: ",len(train_loader))


model = timm.create_model('resnet50', pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=3, bias=True)

from torchsummary import summary
model.to(device=DEVICE, dtype=torch.float)
summary(model, (3, 224, 224))

avg_train_losses = []  # losses of all training epochs
avg_valid_losses = []  # losses of all training epochs
avg_valid_DS = []  # all training epochs

NUM_EPOCHS = 10
LEARNING_RATE = 0.0001


def train_fn(loader_train, model, optimizer, loss_fn1, scaler):
    train_losses = []  # loss of each batch
    valid_losses = []  # loss of each batch

    loop = tqdm(loader_train)
    for batch_idx, (data, label) in enumerate(loop):
        data = data.to(device=DEVICE, dtype=torch.float)
        label = label.to(device=DEVICE, dtype=torch.long)
        # forward
        with torch.cuda.amp.autocast():
            out1 = model(data)
            loss1 = loss_fn1(out1, label)
        # backward
        loss = loss1
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        train_losses.append(loss.item())



    train_loss_per_epoch = np.average(train_losses)
    ## all epochs
    avg_train_losses.append(train_loss_per_epoch)

    return train_loss_per_epoch


def save_checkpoint(state, filename="Resnet_1.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)



loss_func = torch.nn.CrossEntropyLoss()
epoch_len = len(str(NUM_EPOCHS))


def main():
    model.to(device=DEVICE, dtype=torch.float)
    loss_fn1 = loss_func
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn1, scaler)

        print_msg = (f'[{epoch:>{epoch_len}}/{NUM_EPOCHS:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' )

        print(print_msg)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)


if __name__ == "__main__":
    main()
