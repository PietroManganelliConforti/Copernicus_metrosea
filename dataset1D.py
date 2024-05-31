import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

class LSTMSequencePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim,  output_length, num_layers=1):
        super(LSTMSequencePredictor, self).__init__()
        
        self.output_length = output_length
        
        # Initialize the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Initialize the fully connected layer
        self.fc = nn.Linear(hidden_dim, output_length)
    
    def forward(self, x):
        # Set initial hidden and cell states to zeros
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # We only need the output of the last time step
        out = out[:, -1, :]
        
        # Flatten the output from the LSTM layer (batch_size, num_signals * hidden_dim)
        out = out.reshape(out.size(0), -1)
        
        # Pass through the fully connected layer to get the final output of length 7
        out = self.fc(out)
        
        return out

class zipDatasets(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2, dataset3):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.datasets = [dataset1, dataset2, dataset3]

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index], self.dataset3[index]
        
    def __len__(self):
        return min(len(d) for d in self.datasets)

class Dataset_1D_copernicus(torch.utils.data.Dataset):

    def __init__(self, csv_file, window_size, output_size, step=7):
        self.data = pd.read_csv(csv_file) #remove the first row
        self.window_size = window_size
        self.output_size = output_size
        self.step = step

        
    def __len__(self):
        return ((len(self.data)-(self.output_size+self.window_size))//self.step) 

    def __getitem__(self, idx):
        start_idx = idx * self.step
        end_idx = start_idx + self.window_size
        label_end_idx = end_idx + self.output_size

        window_data = self.data.iloc[start_idx:end_idx, 4].values
        window_data = torch.tensor(window_data, dtype=torch.float)

        outputs_data = self.data.iloc[end_idx : label_end_idx, 4].values
        outputs_data = torch.tensor(outputs_data, dtype=torch.float)

        return window_data, outputs_data




class merge_1D_dataset(torch.utils.data.Dataset):

    def __init__(self, folder_path = "dataset_copernicus2/", pred_label_lat = "45.60", 
                 pred_label_lon = "13.54", depth = "1", transforms = None):

        self.folder_path = folder_path
        self.sample_folders = os.listdir(folder_path)
        self.dataset_1D = []
        self.pred_label_lat = pred_label_lat
        self.pred_label_lon = pred_label_lon
        self.depth = depth

        self.transforms = transforms

        self.label_dataset_idx = None

        idx = -1

        for path in self.sample_folders:

            for file in os.listdir(folder_path+path):

                csv_file_path = folder_path+path+"/"+file

                self.dataset_1D.append(Dataset_1D_copernicus(csv_file_path, window_size=30, output_size=7, step=7))

                idx += 1
                
                if (self.pred_label_lat in file) and (self.pred_label_lon in file) and (file[0] == self.depth):

                    self.label_dataset_idx = idx


        
    def __len__(self):
        return len(self.dataset_1D[0]) 

    def __getitem__(self, idx):

        list_of_inputs = []

        labels = None
        
        for i,dataset in enumerate(self.dataset_1D):

            elem = dataset[idx]

            list_of_inputs.append(elem[0])
            
            if self.label_dataset_idx == i:
                
                labels = elem[1]

        output_tensor = torch.stack(list_of_inputs)

        if self.transforms:
            
            for i in range(output_tensor.shape[0]):
                output_tensor[i] = self.transforms(output_tensor[i])
        output_tensor=torch.swapaxes(output_tensor,0,1) #LSTM needs sequence first,then input size
        return output_tensor, labels


def main(): 

    device = "cpu"

    if torch.cuda.is_available():

        device = ("cuda:0")  # Fixed the device to cuda 0

        num_devices = torch.cuda.device_count()

        current_device = torch.cuda.current_device()

        device_name = torch.cuda.get_device_name(current_device) 

        print("Device name:", device_name, "Device number:", current_device, "Number of devices:", num_devices)
        

    os.environ["CUDA_VISIBLE_DEVICES"] = device

    print("Actual device: ", device)
 
    # Dataset parameters:

    csv_file = "dataset_copernicus2/1Ddataset_45.60_13.54/1_45.60_13.54.csv"

    window_size = 30

    output_size = 7

    dataset  = Dataset_1D_copernicus(csv_file, window_size, output_size, step=7)


    # Explore the dataset
    print("Dataset length: ", len(dataset))
    print("Dataset sample: ", dataset[0][1])

    
    # Merge datasets

    dataset_merged_1d = merge_1D_dataset(folder_path = "dataset_copernicus2/", pred_label_lat = "45.60", pred_label_lon = "13.54", depth = "1")

    print("Merged dataset length: ", len(dataset_merged_1d))
    print("Merged dataset sample shape and sample: ", dataset_merged_1d[0][0].shape, dataset_merged_1d[0][0])
    print("Merged dataset label shape and label: ", dataset_merged_1d[0][1].shape, dataset_merged_1d[0][1])


if __name__ == '__main__':
    
    main()
