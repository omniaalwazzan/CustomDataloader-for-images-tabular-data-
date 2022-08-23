# CustomDataloader-for-images-tabular-data-

![This is an image showing images printed from the dataloader](dataloader-images-in-csv/testing_dataloader.png)
# This for loop will print the images from the dataloader (as shown in the image above)
#b=iter(train_loader)

#image, lable=b.next()


#labels_map = {
    0: "Rock",
    1: "Scissors",
    2: "Paper",
#}
#figure = plt.figure(figsize=(8, 8))
#cols, rows = 3, 3
#for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(tr_data), size=(1,)).item()
    img, label = tr_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
