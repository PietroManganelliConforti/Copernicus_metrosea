import os
import cv2
import torch
import numpy as np
import pandas as pd

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

    csv_file = "filtered_datasets_copericus_csv/1Ddataset_45.60_13.54/1_45.60_13.54.csv"

    window_size = 168

    output_size = 24

    dataset  = Dataset_1D_copernicus(csv_file, window_size, output_size)


    # Explore the dataset
    print("Dataset length: ", len(dataset))
    print("Dataset sample: ", dataset[0])


    



if __name__ == '__main__':
    
    main()