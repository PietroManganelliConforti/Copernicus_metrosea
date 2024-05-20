import os
import cv2
import torch
import numpy as np
import pandas as pd
from dataset2D import Dataset_2D_copernicus, merge_2D_dataset, fused_resnet
torch.backends.cudnn.benchmark = True
import time
#torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader, Subset


if __name__ == "__main__":

    device = "cpu"

    if torch.cuda.is_available():

        device = ("cuda:0")  # Fixed the device to cuda 0

        num_devices = torch.cuda.device_count()

        current_device = torch.cuda.current_device()

        device_name = torch.cuda.get_device_name(current_device)


    #load the dataset

    dataset_2D = merge_2D_dataset(folder_path = "2D_Dataset_copernicus_only_tensors/",
                                    label_lat = "45.60", label_lon = "13.54")

    print("dataset: ",  dataset_2D[0][0].shape, dataset_2D[0][1].shape)

    ##mean and variance of the labels:

    mean=np.asarray([16.3281, 16.3271, 16.3276, 16.3307, 16.3349, 16.3392, 16.3471])
    std=np.asarray([5.9852, 5.9800, 5.9789, 5.9768, 5.9834, 5.9868, 5.9880])
    mean_pt =torch.from_numpy(mean).to(device)
    std_pt =torch.from_numpy(std).to(device)


    #split the dataset

    split = 0.8

    split_index = int(split * len(dataset_2D))

    overlapping_indexes = int((30+7)/7)  # where 30 is the window size and 7 is the output size and the stride size. With math.ceil the days overlap is removed

    train_indices = list(range(split_index - overlapping_indexes))
    test_indices = list(range(split_index, len(dataset_2D)))

    train_dataset = Subset(dataset_2D, train_indices)
    test_dataset = Subset(dataset_2D, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)




    #train the model

    fused_resnet_model = fused_resnet()

    fused_resnet_model.to(device)
    #fused_resnet_model = torch.compile(fused_resnet_model)
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(fused_resnet_model.parameters(), lr=0.001)

    num_epochs = 100

    start = time.time()

    print("Training the model...")
    

    for epoch in range(num_epochs):
        start_time = time.time()

        fused_resnet_model.train()

        acc_loss = 0
        label_list = []
        for i, (data, labels) in enumerate(train_dataloader):
            data = data.to(device)
            labels = labels.to(device)
            labels=((labels-mean_pt)/std_pt).float()
            optimizer.zero_grad()

            outputs = fused_resnet_model(data)

            loss = criterion(outputs, labels)
            chunks = list(torch.chunk(labels, labels.size(0), dim=0))
            label_list.extend(chunks)
            loss.backward()

            optimizer.step()
            acc_loss +=loss #(loss*std_pt)+mean_pt ##showing in denormalized, original values

        # Concatenate all items into a single tensor along the first dimension
#        concatenated_items = torch.cat(label_list, dim=0)

        # Calculate mean and standard deviation of the concatenated tensor
#        mean_items = torch.mean(concatenated_items, dim=0)
#        std_items = torch.std(concatenated_items, dim=0)

#        print("Mean of items:", mean_items)
#        print("Standard deviation of items:", std_items)

        epoch_elapsed_time = time.time() - start_time
        
        print("Epoch: ",epoch, "Loss in original space: ", torch.mean(acc_loss)/len(train_dataloader))

    print("Total time: ", time.time()-start)


    #test the model

    fused_resnet_model.eval()

    acc_loss = 0

    with torch.no_grad():

        for i, (data, labels) in enumerate(test_dataloader):

            data = data.to(device)
            labels = labels.to(device)
            labels=labels # ((labels-mean_pt)/std_pt).float() #normalizing labels. If I want to get actual error, need to disable this and enable the other line

            outputs = fused_resnet_model(data)*std_pt+mean_pt

            loss = criterion(outputs, labels)
            acc_loss += loss 
    
    print("Total test loss in original space: ", torch.mean(acc_loss)/len(test_dataloader))
