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
from utils import *

warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter




def test(model, test_dataloader, mean, std, mean_pt, std_pt, device, criterion_print, batch_size):
    #test the model

    model.eval()

    acc_loss = 0

    distance_per_label = torch.zeros(7)

    with torch.no_grad():

        for i, (data, labels) in enumerate(test_dataloader):

            data = data.to(device)
            data = (data - mean)/std

            labels = labels.to(device)

            outputs = model(data) *std_pt+mean_pt

            loss = criterion_print(outputs, labels)
            acc_loss += loss 

            for i in range(7): # così non uso la batchsize

                distance_per_label[i] = distance_per_label[i] + torch.mean(outputs[:,i]-labels[:,i],dim=0)

            ret_value1 = torch.mean(acc_loss).cpu().numpy()/len(test_dataloader)
            ret_value2 = distance_per_label/len(test_dataloader)

    return ret_value1, ret_value2




def run_single_training_and_test(repetition_path, args):

    ret_dict = {}

    random_number = random.randint(1, 1000)
    dir_name = repetition_path # f"runs/{random_number}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory {dir_name} created.")

    writer = SummaryWriter(dir_name)
    ret_dict["tb_dir_name"] = dir_name




    device = "cpu"

    if torch.cuda.is_available() and not args.use_cpu:

        device = ("cuda:0")  # Fixed the device to cuda 0

        num_devices = torch.cuda.device_count()

        current_device = torch.cuda.current_device()

        device_name = torch.cuda.get_device_name(current_device)

    print("Device: ", device)


    ret_dict["loss_alpha"] = args.loss_alpha
    ret_dict["ema_alpha"] = args.ema_alpha

    augmentations = get_augmentation(args.augmentations)


    dataset_2D = merge_2D_dataset(folder_path = "2D_Dataset_copernicus_only_tensors/",
                                    pred_label_lat = "45.60", pred_label_lon = "13.54",
                                    transforms = augmentations)

    print("dataset: ",  dataset_2D[0][0].shape, dataset_2D[0][1].shape)




    #split the dataset

    split = 0.8

    split_index = int(split * len(dataset_2D))

    #TODO CROSS VALIDATION SULLO SPLIT INDEX

    overlapping_indexes = int((30+7)/7)  # where 30 is the window size and 7 is the output size and the stride size. With math.ceil the days overlap is removed

    train_indices = list(range(split_index - overlapping_indexes))
    test_indices = list(range(split_index, len(dataset_2D)))

    train_dataset = Subset(dataset_2D, train_indices)
    test_dataset = Subset(dataset_2D, test_indices)

    batch_size = args.batch_size
    batch_size_test = args.batch_size_test

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False,num_workers=8)

    ret_dict["batch_size"] = batch_size
    ret_dict["batch_size_test"] = batch_size_test

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

    small_net_flag = args.small_net_flag

    if small_net_flag: print("Using the small network, be sure to select the right dataset")

    ret_dict["small_net_flag"] = small_net_flag
    
    fused_resnet_model = fused_resnet(small_net_flag=small_net_flag)
    #fused_resnet_model = fused_resnet_LSTM()

    fused_resnet_model.to(device)
    #fused_resnet_model = torch.compile(fused_resnet_model)

    criterion = nn.MSELoss()
    if args.loss_alpha != 0.0:
        criterion = CustomLoss(alpha=args.loss_alpha)

    criterion_print= torch.nn.L1Loss()

    lr = args.lr

    optimizer = torch.optim.Adam(fused_resnet_model.parameters(), lr = lr)

    ret_dict["lr"] = lr

    num_epochs = args.num_epochs

    start = time.time()

    print("Training the model...")

    #print("Start time: ", time.strftime("%H:%M:%S", time.gmtime(start)))
    best_model = None
    best_loss = float('inf')
    best_model_ema = None
    best_loss_ema = float('inf')

    #additional code for SWA
    ema_model = emaodel(fused_resnet_model, alpha=args.ema_alpha,device=device)
    #swa_model = torch.optim.swa_utils.AveragedModel(fused_resnet_model).to(device)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    #swa_start = 2
    #swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.05)

    for epoch in range(num_epochs):

        start_time = time.time()


        acc_loss = 0
        train_acc_loss = 0
        label_list = []



        fused_resnet_model.train()
        for i, (data, labels) in enumerate(train_dataloader):

            data = data.to(device)

            data = (data - mean)/std
            labels = labels.to(device)
            labels=((labels-mean_pt)/std_pt).float()

            batch, seq, channels, w, h = data.shape

            optimizer.zero_grad()
            outputs = fused_resnet_model(data)

            loss = criterion(outputs, labels)
            loss_unorm = criterion_print(outputs*std_pt+mean_pt, labels*std_pt+mean_pt)
            loss.backward()

            optimizer.step()
            ema_model.update_ema_weights()
            acc_loss +=loss_unorm 
            train_acc_loss += loss

        L1_train_loss=(torch.mean(acc_loss).detach().cpu().numpy()/len(train_dataloader))
        train_loss=(torch.mean(train_acc_loss).detach().cpu().numpy()/len(train_dataloader))
        epoch_elapsed_time = time.time() - start_time
        print("Epoch:",epoch, "L1 training Loss in original space: ", L1_train_loss, " train Loss: ", train_loss)
        torch.cuda.empty_cache()

        if epoch > -1:
            #test evaluated at the beginning of the epoch
            actual_test_loss,distance_label_loss=test(fused_resnet_model, test_dataloader, mean, std, mean_pt, std_pt, device, criterion_print,batch_size_test)
            print("Epoch:",epoch, "    Test L1 Loss in original space: ", actual_test_loss)
            #print("Distance per label: ", distance_label_loss)
            ## evaluating EMA model
            ema_model.eval()
            actual_test_loss_ema,distance_label_loss_ema=test(ema_model, test_dataloader, mean, std, mean_pt, std_pt, device, criterion_print,batch_size_test)
            print("Epoch:",epoch, "    EMA     Loss in original space: ", actual_test_loss_ema)

            if actual_test_loss < best_loss:
                best_loss = actual_test_loss
                best_model_wts = fused_resnet_model.state_dict()
                ret_dict["test_loss"] = best_loss
                ret_dict["mean_distance"] = distance_label_loss.tolist()
                ret_dict["train_loss"] = train_loss
                ret_dict["L1_train_loss"] = L1_train_loss
                ret_dict["epoch"] = epoch

            if actual_test_loss_ema < best_loss_ema:
                best_loss_ema = actual_test_loss_ema
                best_model_wts_ema = ema_model.state_dict()
                ret_dict["test_loss_ema"] = best_loss_ema
                ret_dict["mean_distance_ema"] = distance_label_loss_ema.tolist()
                ret_dict["epoch_ema"] = epoch

            writer.add_scalars('data/', {'L1_training_loss':L1_train_loss,'L1_test_loss':actual_test_loss,'EMA_test_loss':actual_test_loss_ema}, global_step=epoch)
            writer.flush()

    ema_model.load_state_dict(best_model_wts_ema)
#    ema_model.update_bn_statistics(train_dataloader)
#    actual_test_loss_ema_BN,distance_label_loss_ema=test(ema_model, test_dataloader, mean, std, mean_pt, std_pt, device, criterion_print,batch_size_test)
    print("The best test loss is:", best_loss)
    print("The best EMA  loss is:", best_loss_ema)
#    print("Corrected EMA loss is:", actual_test_loss_ema_BN)

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 10))

    # Plot for the first six samples in the subplots
    for i, ax in enumerate(axes.flat):
        plot_label_vs_prediction(ax, sample_idx=i, fused_resnet_model=fused_resnet_model, 
                                 best_model_wts=best_model_wts, test_dataset=test_dataset,
                                 mean=mean, std=std, mean_pt=mean_pt, std_pt=std_pt, device=device)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    plt.savefig(os.path.join(repetition_path,"label_vs_prediction_plot.png"))


    writer.add_figure('label_vs_prediction_plot', fig, global_step=0)
    writer.flush()
    writer.close()
    # Show the plot
    plt.show()

    print("saved the plot")

    return ret_dict


if __name__ == "__main__":

    #parse --get only tensor flag
    parser = argparse.ArgumentParser(description='run training')
    parser.add_argument('--batch_size', type=int,default=64, help='batch size for training')
    parser.add_argument('--batch_size_test', type=int,default=256, help='batch size for training')
    #parser.add_argument('--alpha', type=float, default=0.0, help=' custom loss regularization term') #0.0 means no custom loss
    parser.add_argument('--loss_alpha', type=float, default=0.0, help=' custom loss regularization term') #0.0 means no custom loss
    parser.add_argument('--ema_alpha', type=float, default=0.9, help=' exponential moving average alpha')
    parser.add_argument('--use_cpu', action='store_true', help='use CPU')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--small_net_flag', action='store_true', help='use small network')
    parser.add_argument('--augmentations', type=int, default=0, help='use augmentations')
    parser.add_argument('--test_name', type=str, default="test", help='name of the test')
    
    args = parser.parse_args()

    repetitions = 5

    repetitions_dict = {}

    if args.test_name == "test":

        test_path = "results/test_"+time.strftime("%Y%m%d-%H%M%S")+"/"

    else:
            
        test_path = "results/test_"+args.test_name+"/"

    if not os.path.exists(test_path):
            
        os.makedirs(test_path)
        print("Created folder: ", test_path)

    for i in range(repetitions):

        repetition_path = test_path+ "repetition_"+str(i)+"/"

        if not os.path.exists(repetition_path):
            
            os.makedirs(repetition_path)
            print("Created folder: ", repetition_path)

        repetitions_dict["repetition_"+str(i)] = run_single_training_and_test(repetition_path, args)

    


    results_dict = {}
    results_dict["final_results"] = {}

    #save the mean results of test loss and train loss and L1 train loss in the repetitions_dict["final_results"]
    
    results_dict["final_results"]["mean_test_loss"] = 0
    results_dict["final_results"]["mean_train_loss"] = 0
    results_dict["final_results"]["mean_L1_train_loss"] = 0
    results_dict["final_results"]["mean_distance"] = 0

    for i in range(repetitions):
        results_dict["final_results"]["mean_test_loss"] += repetitions_dict["repetition_"+str(i)]["test_loss"]
        results_dict["final_results"]["mean_train_loss"] += repetitions_dict["repetition_"+str(i)]["train_loss"]
        results_dict["final_results"]["mean_L1_train_loss"] += repetitions_dict["repetition_"+str(i)]["L1_train_loss"]
        results_dict["final_results"]["mean_distance"] = np.add(results_dict["final_results"]["mean_distance"], repetitions_dict["repetition_"+str(i)]["mean_distance"])

    results_dict["final_results"]["mean_test_loss"] = results_dict["final_results"]["mean_test_loss"]/repetitions
    results_dict["final_results"]["mean_train_loss"] = results_dict["final_results"]["mean_train_loss"]/repetitions
    results_dict["final_results"]["mean_L1_train_loss"] = results_dict["final_results"]["mean_L1_train_loss"]/repetitions
    results_dict["final_results"]["mean_distance"] = results_dict["final_results"]["mean_distance"]/repetitions



    #add other args
    args_dict = vars(args)

    args_list = [key for key in args_dict if not key.startswith("__")]

    results_dict["final_results"].update({arg: getattr(args, arg) for arg in args_list})

    #save dict in test path as dict.json. print in a readable way

    pd.DataFrame(repetitions_dict).to_json(test_path+"run_report.json", indent=4)

    pd.DataFrame(results_dict).to_json(test_path+"results.json", indent=4)
    
    print("Saved the results in the folder: ", test_path)

    #print("Final results: ", repetitions_dict["final_results"])



    
