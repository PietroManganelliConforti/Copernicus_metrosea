import os
import cv2
import torch
import numpy as np
import pandas as pd
from dataset2D import Dataset_2D_copernicus, merge_2D_dataset, fused_resnet



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


    #dataloader

    split = 0.8

    train_size = int(split * len(dataset_2D))
    test_size = len(dataset_2D) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset_2D, [train_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)



    #dataloader2D = torch.utils.data.DataLoader(dataset_2D, batch_size=64, shuffle=True)




    #train the model

    fused_resnet_model = fused_resnet()

    fused_resnet_model.to(device)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(fused_resnet_model.parameters(), lr=0.001)

    num_epochs = 100

    import time

    start = time.time()

    print("Training the model", start)

    for epoch in range(num_epochs):
            
        fused_resnet_model.train()

        loss = 0

        for i, (data, labels) in enumerate(train_dataloader):

            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = fused_resnet_model(data)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            loss += loss.item()

        
        print("Epoch: ", epoch, "Loss: ", loss/len(train_dataloader))
    

    print("Total time: ", time.time()-start)


    #test the model

    fused_resnet_model.eval()

    acc_loss = 0

    with torch.no_grad():

        for i, (data, labels) in enumerate(test_dataloader):

            data = data.to(device)
            labels = labels.to(device)

            outputs = fused_resnet_model(data)

            loss = criterion(outputs, labels)

            acc_loss += loss.item()
    
    print("Total test loss: ", acc_loss/len(test_dataloader))