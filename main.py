import os
import cv2
import torch
import numpy as np
import pandas as pd
from dataset2D import Dataset_2D_copernicus, merge_2D_dataset, fused_resnet, fused_resnet_LSTM
torch.backends.cudnn.benchmark = True
import time
import matplotlib.pyplot as plt
#torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import argparse
import random
import warnings
warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter



class CustomLoss(nn.Module):
    def __init__(self, alpha=0.0):

        super(CustomLoss, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha))  # Regularization strength
        
    def forward(self, predictions, targets):
        """
        Forward pass of the loss function.
            
        Returns:
            loss (torch.Tensor): Combined loss value.
        """
        # Compute MSE loss
        mse_loss = nn.functional.mse_loss(predictions, targets)
        
        # Compute regularization term (Total Variation Regularization)
        tv_loss = self.total_variation_regularization(predictions)
        
        # Combine MSE loss with regularization term
        total_loss = mse_loss + self.alpha * tv_loss
        
        return total_loss
    
    def total_variation_regularization(self, predictions):
        """
        Computes the Total Variation regularization term.
        """
        #the value has to have the gradient
        
        tv_loss = torch.tensor(0.0, requires_grad=True).to(predictions.device)

        torch.tensor(0.0, requires_grad=True).to(predictions.device)

        # Compute the total variation for each channel
        for i in range(predictions.size(1) - 1):  # Loop until the second last item
            # Compute the absolute difference between adjacent values along the sequence dimension
            diff = torch.diff(predictions[:, i + 1] - predictions[:, i])
            diff = abs(diff)
            # Sum over the output dimension and add to the regularization loss, keep trak the gradient
            diff = torch.sum(diff)

            tv_loss = tv_loss + diff

        return tv_loss



def test(model, test_dataloader, mean, std, mean_pt, std_pt, device, criterion_print,batch_size):
    #test the model

    model.eval()

    acc_loss = 0

    distance_per_label = torch.zeros(7)

    with torch.no_grad():

        for i, (data, labels) in enumerate(test_dataloader):

            data = data.to(device)
            data = (data - mean)/std
            # batch, seq, channels, w, h = data.shape

            # data = data.view(-1, data.shape[2],data.shape[3],data.shape[4])

            # data = F.interpolate(data, size=(224,224),mode='bilinear',align_corners=False)
            # data = data.view(batch,seq,channels,224,224)


            labels = labels.to(device)
            labels=labels # ((labels-mean_pt)/std_pt).float() #normalizing labels. If I want to get actual error, need to disable this and enable the other line

            outputs = model(data) *std_pt+mean_pt

            loss = criterion_print(outputs, labels)
            acc_loss += loss 

            for i in range(7): # così non uso la batchsize

                distance_per_label[i] = distance_per_label[i] + torch.mean(outputs[:,i]-labels[:,i],dim=0)

            ret_value1 = torch.mean(acc_loss).cpu().numpy()/len(test_dataloader)
            ret_value2 = distance_per_label/len(test_dataloader)

    return ret_value1, ret_value2




def get_mean_and_std(dataset_2D, monodim = True):
    
    mean =0
    std = 0

    if not monodim:
        mean = torch.zeros((16, 3))
        std = torch.zeros((16, 3))


    for images,_ in dataset_2D:

        if monodim:
            mean += images.mean()  
            std += images.std()   
        else: 
            mean += images.mean(dim=[2, 3])  
            std += images.std(dim=[2, 3])   



    # Divide by the total number of images to get the mean and std for the entire dataset
    mean /= len(dataset_2D)
    std /= len(dataset_2D)

    if not monodim:
        mean = mean.unsqueeze(-1).unsqueeze(-1)
        std = std.unsqueeze(-1).unsqueeze(-1)
        print("mean and std shape and mean values: ", mean.shape, mean.mean(), std.shape, std.mean())
    else:
        print("mean and std: ", mean, std)
    
    return mean, std



def plot_label_vs_prediction(ax, sample_idx, fused_resnet_model, best_model_wts, test_dataset, mean, std, mean_pt, std_pt, device):
    
    with torch.no_grad():
        input_data, label = test_dataset[sample_idx]
        input_data, label = input_data.to(device), label.to(device)

        input_data = input_data.unsqueeze(0)  # Add batch dimension if needed
        input_data = ((input_data - mean) / std).float()
        batch, seq, channels, w, h = input_data.shape

        # input_data = input_data.view(-1, input_data.shape[2],input_data.shape[3],input_data.shape[4])
        # input_data = F.interpolate(input_data, size=(224,224),mode='bilinear',align_corners=False)
        # input_data = input_data.view(batch,seq,channels,224,224)

        fused_resnet_model.load_state_dict(best_model_wts)
        fused_resnet_model.eval()
        prediction = fused_resnet_model(input_data)
        prediction = (prediction * std_pt + mean_pt).squeeze(0)  # Plotting in the original space

        label = label.cpu().numpy()
        prediction = prediction.cpu().numpy()

        ax.plot(label, 'o-', label='Label', color='blue')
        ax.plot(prediction, 'x-', label='Prediction', color='red')
        
        #print from zero

        ax.set_ylim(20)

        ax.set_title('Label vs Prediction')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()






if __name__ == "__main__":

    random_number = random.randint(1, 1000)
    dir_name = f"runs/{random_number}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory {dir_name} created.")

    writer = SummaryWriter(dir_name)

    device = "cpu"

    if torch.cuda.is_available():

        device = ("cuda:0")  # Fixed the device to cuda 0

        num_devices = torch.cuda.device_count()

        current_device = torch.cuda.current_device()

        device_name = torch.cuda.get_device_name(current_device)


    #parse --get only tensor flag
    parser = argparse.ArgumentParser(description='run training')
    parser.add_argument('--batch_size', type=int,default=8, help='batch size for training')
    parser.add_argument('--batch_size_test', type=int,default=256, help='batch size for training')
    parser.add_argument('--alpha', type=float, default=0.0, help='alpha for the regularization term of the custom loss')
    
    
    args = parser.parse_args()

    
    augmentations = transforms.Compose([    
        #transforms.ElasticTransform(alpha=5.0, sigma=0.5) #alpha, sigma sono quelle di default/10
    ])
    
    dataset_2D = merge_2D_dataset(folder_path = "2D_Dataset_copernicus_only_tensors/",
                                    label_lat = "45.60", label_lon = "13.54",
                                    transforms = augmentations)

    print("dataset: ",  dataset_2D[0][0].shape, dataset_2D[0][1].shape)




    #split the dataset

    split = 0.8

    split_index = int(split * len(dataset_2D))

    overlapping_indexes = int((30+7)/7)  # where 30 is the window size and 7 is the output size and the stride size. With math.ceil the days overlap is removed

    train_indices = list(range(split_index - overlapping_indexes))
    test_indices = list(range(split_index, len(dataset_2D)))

    train_dataset = Subset(dataset_2D, train_indices)
    test_dataset = Subset(dataset_2D, test_indices)

    batch_size = args.batch_size
    batch_size_test = args.batch_size_test

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False,num_workers=8)




    #mean and std of the labels

    mean_pt = np.asarray([16.3281, 16.3271, 16.3276, 16.3307, 16.3349, 16.3392, 16.3471])
    std_pt = np.asarray([5.9852, 5.9800, 5.9789, 5.9768, 5.9834, 5.9868, 5.9880])

    mean_pt = torch.from_numpy(mean_pt).to(device)
    std_pt = torch.from_numpy(std_pt).to(device)


    #get mean and std of the dataset

    mean, std = get_mean_and_std(train_dataset)

    # approximate mean and std 

    # mean = torch.tensor([2.9796, 2.9842, 2.9871])
    # std = torch.tensor([7.2536, 7.2696, 7.2694])

    mean = mean.to(device)
    std = std.to(device)





    #train the model

    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True  # this is to avoid the error of torch.hub.load

    small_net_flag = False

    if small_net_flag: print("Using the small network, be sure to select the right dataset")
    
    fused_resnet_model = fused_resnet(small_net_flag=small_net_flag)

    fused_resnet_model.to(device)
    #fused_resnet_model = torch.compile(fused_resnet_model)

    criterion = nn.MSELoss()
    if args.alpha > 0.0:
        criterion = CustomLoss(alpha=args.alpha)

    criterion_print= torch.nn.L1Loss()

    optimizer = torch.optim.Adam(fused_resnet_model.parameters(), lr=2e-4)

    num_epochs = 30

    start = time.time()

    print("Training the model...")

    print("Start time: ", time.strftime("%H:%M:%S", time.gmtime(start)))
    best_model = None
    best_loss = float('inf')



    for epoch in range(num_epochs):

        start_time = time.time()

        fused_resnet_model.train()

        acc_loss = 0
        train_acc_loss = 0
        label_list = []


        actual_test_loss,distance_label_loss=test(fused_resnet_model, test_dataloader, mean, std, mean_pt, std_pt, device, criterion_print,batch_size_test)

        for i, (data, labels) in enumerate(train_dataloader):

            data = data.to(device)

            data = (data - mean)/std
            labels = labels.to(device)
            labels=((labels-mean_pt)/std_pt).float()

            batch, seq, channels, w, h = data.shape

            # data = data.view(-1, data.shape[2],data.shape[3],data.shape[4])
            # data = F.interpolate(data, size=(224,224),mode='bilinear',align_corners=False)
            # data = data.view(batch,seq,channels,224,224)

            optimizer.zero_grad()
            outputs = fused_resnet_model(data)

            loss = criterion(outputs, labels)
            loss_unorm = criterion_print(outputs*std_pt+mean_pt, labels*std_pt+mean_pt)
            #chunks = list(torch.chunk(labels, labels.size(0), dim=0))
            #label_list.extend(chunks)
            loss.backward()

            optimizer.step()
            acc_loss +=loss_unorm 
            train_acc_loss += loss
            
        L1_train_loss=(torch.mean(acc_loss).detach().cpu().numpy()/len(train_dataloader))
        train_loss=(torch.mean(train_acc_loss).detach().cpu().numpy()/len(train_dataloader))
        epoch_elapsed_time = time.time() - start_time
        print("Epoch:",epoch, "L1 training Loss in original space: ", L1_train_loss, " train Loss: ", train_loss)
        torch.cuda.empty_cache()

        if epoch > -1:
            #test evaluated at the beginning of the epoch

            print("Epoch:",epoch, "    Test L1 Loss in original space: ", actual_test_loss)
            print("Distance per label: ", distance_label_loss)

            if actual_test_loss < best_loss:
                best_loss = actual_test_loss
                best_model_wts = fused_resnet_model.state_dict()
            writer.add_scalars('data/', {'L1_training_loss':L1_train_loss,'L1_test_loss':actual_test_loss}, global_step=epoch)
            writer.flush()

#    actual_test_loss=test(fused_resnet_model, test_dataloader, mean, std, mean_pt, std_pt, device, criterion_print,batch_size_test)
    
#    if actual_test_loss < best_loss:
#        best_loss = actual_test_loss
#        best_model_wts = fused_resnet_model.state_dict()

print("The best test loss is:", best_loss)

# Create a figure and a set of subplots
fig, axes = plt.subplots(3, 3, figsize=(20, 10))

# Plot for the first six samples in the subplots
for i, ax in enumerate(axes.flat):
    plot_label_vs_prediction(ax, sample_idx=i, fused_resnet_model=fused_resnet_model, best_model_wts=best_model_wts, test_dataset=test_dataset, mean=mean, std=std, mean_pt=mean_pt, std_pt=std_pt, device=device)

# Adjust layout to prevent overlap
plt.tight_layout()

plt.savefig(os.path.join(dir_name,"six_samples_label_vs_prediction.png"))
writer.add_figure('six_samples_label_vs_prediction', fig, global_step=0)
writer.flush()
writer.close()
# Show the plot
plt.show()

print("saved the plot")
