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
from augmentations import get_augmentation
import torch.nn.functional as F  
import torch.nn as nn
import argparse
import random
import warnings
import copy

warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter


class EMAModel(nn.Module):
    def __init__(self, model, alpha=0.99, device='cpu',bn=False):
        super(EMAModel, self).__init__()
        self.model = model
        self.alpha = alpha
        self.device = device
        self.model.to(self.device)
        self.ema_weights = {name: param.clone().detach().to(self.device) for name, param in model.named_parameters() if param.requires_grad}
        self.bn = bn
        if self.bn:
            self.ema_buffers = {name: buf.clone().detach().to(self.device) for name, buf in model.named_buffers()}

    def update_ema_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_weights[name].mul_(self.alpha).add_(param.data.to(self.device), alpha=1 - self.alpha)
        if self.bn:
            for name, buf in self.model.named_buffers():
                if "running_mean" in name or "running_var" in name:
                    self.ema_buffers[name].mul_(self.alpha).add_(buf.to(self.device), alpha=1 - self.alpha)

    def forward(self, inputs):
        # Create a copy of the model to apply EMA weights
        ema_model = copy.deepcopy(self.model)
        ema_model.to(self.device)

        # Apply EMA weights to the copied model
        for name, param in ema_model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.ema_weights[name])
        if self.bn:
            for name, buf in ema_model.named_buffers():
                if "running_mean" in name or "running_var" in name:
                    buf.copy_(self.ema_buffers[name])

        # Perform inference
        ema_model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = ema_model(inputs)

        return outputs

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


        #ax.set_ylim(20)

        ax.set_title('Label vs Prediction')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()






