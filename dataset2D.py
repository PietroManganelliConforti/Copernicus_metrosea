import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch import nn

class Dataset_2D_copernicus(torch.utils.data.Dataset):

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.sample_folders = os.listdir(folder_path)
        self.lat = self.folder_path.split("_")[-2]
        self.lon = self.folder_path.split("_")[-1]

        
    def __len__(self):
        return len(self.sample_folders) 

    def __getitem__(self, idx):
        sample_folder = self.folder_path + "/" + str(idx) + "/"

        tensor_data = torch.load(sample_folder+str(idx)+ "_"+str(self.lat)+ "_"+str(self.lon)+"_morl_3depths_tensor.pt")
        labels = torch.load(sample_folder+str(idx)+ "_"+str(self.lat)+ "_"+str(self.lon)+"_morl_dep1_labels.pt")

        return tensor_data, labels


class merge_2D_dataset(torch.utils.data.Dataset):

    def __init__(self, folder_path = "2D_Dataset_copernicus_only_tensors/",pred_label_lat = "45.60", pred_label_lon = "13.54", transforms = None):

        self.folder_path = folder_path
        self.sample_folders = os.listdir(folder_path)
        
        self.dataset_2D_list = [ Dataset_2D_copernicus(self.folder_path+path) for path in self.sample_folders]

        print("N of 2D dataset loaded: ", len(self.dataset_2D_list))

        self.label_lat = pred_label_lat
        self.label_lon = pred_label_lon

        self.transforms = transforms

        
    def __len__(self):
        return len(self.dataset_2D_list[0]) 

    def __getitem__(self, idx):

        list_of_inputs = []

        labels = None
        
        for i,dataset in enumerate(self.dataset_2D_list):
            elem = dataset[idx]

            list_of_inputs.append(elem[0])
            
            if (dataset.lat == self.label_lat and dataset.lon == self.label_lon):
                
                labels = elem[1]

        output_tensor = torch.stack(list_of_inputs)

        if self.transforms:
            
            for i in range(output_tensor.shape[0]):
                output_tensor[i] = self.transforms(output_tensor[i])

        return output_tensor, labels







def test_2D_dataset_copernicus(device):

    
    dataset_2D = Dataset_2D_copernicus("2D_Dataset_copernicus_only_tensors/2Ddataset_45.60_13.54")
    
    print("len of dataset: ", len(dataset_2D))

    print("dataset[0][0].shape: ", dataset_2D[0][0].shape)

    ##import a pretrained Resnet18 and use it on dataset

    resnet18 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

    #change resnet18 output to 7 values

    resnet18.fc = torch.nn.Linear(in_features=512, out_features=7, bias=True)

    resnet18.to(device)


    #dataloader

    split = 0.8

    train_size = int(split * len(dataset_2D))
    test_size = len(dataset_2D) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset_2D, [train_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    #train the model

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(resnet18.parameters(), lr=0.001)

    num_epochs = 100

    import time

    start = time.time()

    for epoch in range(num_epochs):
            
        resnet18.train()

        loss = 0

        for i, (data, labels) in enumerate(train_dataloader):

            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = resnet18(data)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            loss += loss.item()

        
        print("Epoch: ", epoch, "Loss: ", loss/len(train_dataloader))



    #test the model

    resnet18.eval()

    acc_loss = 0

    with torch.no_grad():
            
        for i, (data, labels) in enumerate(test_dataloader):

            data = data.to(device)
            labels = labels.to(device)

            outputs = resnet18(data)

            loss = criterion(outputs, labels)

            print("Test Loss: ", loss.item())

            acc_loss += loss.item()


    print("Average test loss: ", acc_loss/len(test_dataloader))

    end = time.time()

    print("Time taken: ", end-start)


class fused_resnet(torch.nn.Module):

    def __init__(self, small_net_flag = False):

        super(fused_resnet, self).__init__()

        #create 16 resnet18 and fuse the last layer together
        if small_net_flag:  #https://github.com/chenyaofo/pytorch-cifar-models

            print("Using small networks")

            self.resnet18_list = [torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True, verbose=False) for i in range(16)]

            for i in range(16):

                self.resnet18_list[i].fc = torch.nn.Linear(in_features=64, out_features=7, bias=True)

        else:
            self.resnet18_list = [torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True, verbose=False) for i in range(16)]

            for i in range(16):
                    
                self.resnet18_list[i].fc = torch.nn.Linear(in_features=512, out_features=7, bias=True)

        self.resnet18_list = torch.nn.ModuleList(self.resnet18_list)

        self.fusion_layer = torch.nn.Linear(in_features=16*7, out_features=7, bias=True)

    def forward(self, x):

        outputs = []

        for i in range(16):

            out = self.resnet18_list[i](x[:,i,:,:,:])


            outputs.append(out)

        outputs = torch.stack(outputs, dim = 1)


        outputs = outputs.squeeze()
        if outputs.dim() == 2:
            outputs = outputs.unsqueeze(0)
        outputs=outputs.view(outputs.shape[0],-1)
        #import pdb; pdb.set_trace()
        outputs = self.fusion_layer(outputs)

        return outputs

import torch
import torch.nn as nn

class fused_resnet_LSTM(torch.nn.Module): # da provare con 16 LSTM
    def __init__(self):
        super(fused_resnet_LSTM, self).__init__()

        # Create 16 resnet18 and fuse the last layer together
        self.resnet18_list = [torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True) for _ in range(16)]

        for i in range(16):
            self.resnet18_list[i].fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)

        self.resnet18_list = torch.nn.ModuleList(self.resnet18_list)

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
        
        # Define the final fully connected layer to map LSTM output to 7 predictions
        self.fc = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        outputs = []

        for i in range(16):
            out = self.resnet18_list[i](x[:, i, :, :, :])
            outputs.append(out)

        # Stack outputs and reshape for the LSTM
        outputs = torch.cat(outputs, dim=1).unsqueeze(1)  # Shape: (batch_size, 1, 16)

        # Initialize hidden state and cell state for LSTM
        h0 = torch.zeros(1, x.size(0), 32).to(x.device)
        c0 = torch.zeros(1, x.size(0), 32).to(x.device)

        # Pass the outputs through the LSTM
        lstm_out, _ = self.lstm(outputs, (h0, c0))  # Shape: (batch_size, 1, hidden_size)

        # Process the LSTM output to produce 7 predictions
        predictions = []
        for t in range(7):
            pred = self.fc(lstm_out[:, -1, :])
            predictions.append(pred)

        # Stack the predictions to form the final output
        final_output = torch.cat(predictions, dim=1)  # Shape: (batch_size, 7)

        return final_output



if __name__ == "__main__":

    device = "cpu"

    if torch.cuda.is_available():

        device = ("cuda:0")  # Fixed the device to cuda 0

        num_devices = torch.cuda.device_count()

        current_device = torch.cuda.current_device()

        device_name = torch.cuda.get_device_name(current_device)

        test_2D_dataset_copernicus(device)

        exit(0)

        #print merged dataset

        dataset_2D = merge_2D_dataset(folder_path = "2D_Dataset_copernicus_only_tensors/",
                                      pred_label_lat = "45.60", pred_label_lon = "13.54")


        print("dataset: ",  dataset_2D[0][0].shape, dataset_2D[0][1].shape)

        dummy = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

        dummy.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)

        #set batch norm in eval mode

        for layer in dummy.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()

        dummy_input = dataset_2D[0][0][0].unsqueeze(0)

        print( "dummy(dummy_input) works: ", dummy(dummy_input).shape)



        dataloader2D = torch.utils.data.DataLoader(dataset_2D, batch_size=64, shuffle=True)




        #train the model

        fused_resnet_model = fused_resnet()

        fused_resnet_model.to(device)

        criterion = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(fused_resnet_model.parameters(), lr=0.001)

        num_epochs = 1000

        import time

        start = time.time()

        for epoch in range(num_epochs):
                
            fused_resnet_model.train()

            loss = 0

            for i, (data, labels) in enumerate(dataloader2D):

                data = data.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = fused_resnet_model(data)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                loss += loss.item()

            
            print("Epoch: ", epoch, "Loss: ", loss/len(dataloader2D))

