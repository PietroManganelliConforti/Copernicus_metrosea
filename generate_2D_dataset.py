import argparse
import json
import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange, tqdm
import os
import torch

from dataset1D import Dataset_1D_copernicus, zipDatasets
import pywt
import numpy as np
import matplotlib.pyplot as plt
import shutil

import torch.nn.functional as F



def generate_cwt_from_1Ddataset(data_path, output_path, size = (224, 224), save_dataset_as_images=False):

    wave = 'morl'

    window_size = 30
    output_size = 7
    step = 7

    scales = np.arange(1,window_size+1)


    print("Generating CWT from 1D dataset from: ", data_path)

    for root, dirs, files in os.walk(data_path):

        print("root,dirs and files: ", root, dirs, files)

        if len(files) == 3 :

            file_1, file_3, file_5 = files

            assert(file_1.endswith('.csv'))

            dep1,lat1,lon1 = file_1.split("_")

            lon1 = lon1.split(".csv")[0]

            dep3, lat3, lon3 = file_3.split("_")

            lon3 = lon3.split(".csv")[0]

            dep5, lat5, lon5 = file_5.split("_")

            lon5 = lon5.split(".csv")[0]

            assert(lat1 == lat3 == lat5)
            assert(lon1 == lon3 == lon5)

            lat = lat1
            lon = lon1

            dep_list = [dep1, dep3, dep5]

            folder_name = output_path+"2Ddataset_"+lat+"_"+lon+"/"

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                print("Created folder: ", folder_name)

            print("opening dataset from files: \n", root+"/"+file_1,"\n", root+"/"+file_3, "\n", root+"/"+file_5)

            dataset_1 = Dataset_1D_copernicus(
                csv_file=root+"/"+file_1,
                window_size = window_size,
                output_size = output_size,
                step = step
            )

            dataset_3 = Dataset_1D_copernicus(
                csv_file=root+"/"+file_3,
                window_size= window_size,
                output_size= output_size,
                step = step
            )


            dataset_5 = Dataset_1D_copernicus(
                csv_file=root+"/"+file_5,
                window_size= window_size,
                output_size= output_size,
                step = step
            )

            zippedDataset = zipDatasets(dataset_1, dataset_3, dataset_5)


            def generate_data_from_zipped_dataset(dataset, folder_name, wave, dep_list, lat, lon):

                print("len of datasets: ", len(dataset))

                dep1, dep3, dep5 = dep_list
            
                for i in range(len(dataset)):

                    sample_1, sample_3, sample_5 = dataset[i]

                    sample_folder = folder_name+str(i)+"/"

                    if not os.path.exists(sample_folder):
                        os.makedirs(sample_folder)
                        print("Created folder: ", sample_folder)

                    data_1 = sample_1[0]
                    labels_1 = sample_1[1]

                    data_3 = sample_3[0] 
                    labels_3 = sample_3[1]

                    data_5 = sample_5[0]
                    labels_5 = sample_5[1]

                    coef_1, freqs_1=pywt.cwt(np.array(data_1), scales, wave)
                    coef_3, freqs_3=pywt.cwt(np.array(data_3), scales, wave)
                    coef_5, freqs_5=pywt.cwt(np.array(data_5), scales, wave)

                    stacked_tensor_coef = torch.stack([torch.tensor(coef_1), torch.tensor(coef_3), torch.tensor(coef_5)], dim=0)

                    stacked_tensor_coef = F.interpolate(stacked_tensor_coef.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)


                    print("Stacked tensor shape: ", stacked_tensor_coef.shape)

                    torch.save(stacked_tensor_coef, sample_folder+str(i)+"_"+lat+"_"+lon+"_"+wave+"_3depths_tensor.pt")

                    print("Saved tensor: ", sample_folder+str(i)+"_"+lat+"_"+lon+"_"+wave+"_3depths_tensor.pt", stacked_tensor_coef.shape)

                    
                    def save_coef_image(coef, data, labels, sample_folder, i, dep, lat, lon, wave):

                        sample_name = str(i)+"_"+dep+"_"+lat+"_"+lon+"_"+wave

                        plt.matshow(coef)
                        img_name = sample_name + ".png"

                        plt.matshow(coef)
                        img_name = sample_name + ".png"
                        plt.title("idx: "+str(i)+", dep: "+dep+", lat: "+lat+", lon: "+lon+", wave: "+wave)
                        plt.savefig(sample_folder+img_name, bbox_inches='tight',pad_inches=0.0 )
                        plt.clf()
                        plt.close()

                        ##save also data in the folder as csv
                        txt_name = sample_name + "_1D.csv"
                        np.savetxt(sample_folder+txt_name, data, delimiter=",")

                        label_name = sample_name + "_labels.csv"
                        np.savetxt(sample_folder+label_name, labels, delimiter=",")

                        print("Saved "+str(i)+" image: ", img_name)


                    if save_dataset_as_images:
                        save_coef_image(coef_1, data_1, labels_1, sample_folder, i, dep1, lat, lon, wave)
                        save_coef_image(coef_3, data_3, labels_3, sample_folder, i, dep3, lat, lon, wave)
                        save_coef_image(coef_5, data_5, labels_5, sample_folder, i, dep5, lat, lon, wave)
                    else:
                        torch.save(torch.tensor(labels_1), sample_folder+str(i)+"_"+lat+"_"+lon+"_"+wave+"_dep1_labels.pt")
            


            generate_data_from_zipped_dataset(zippedDataset, folder_name, wave, dep_list, lat, lon)

    return


def generate_data_from_one_dataset(dataset, folder_name, wave, dep_list, lat, lon):

    print("len of dataset: ", len(dataset))

    for i in range(len(dataset)):

        sample = dataset[i]

        sample_folder = folder_name+str(i)+"/"

        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
            print("Created folder: ", sample_folder)

        data = np.array(sample[0])
        labels = np.array(sample[1])

        coef, freqs=pywt.cwt(data, scales, wave)

        plt.matshow(coef)

        sample_name = str(i)+"_"+dep+"_"+lat+"_"+lon+"_"+wave
        
        img_name = sample_name + ".png"

        plt.title("idx: "+str(i)+", dep: "+dep+", lat: "+lat+", lon: "+lon+", wave: "+wave)
        plt.savefig(sample_folder+img_name, bbox_inches='tight',pad_inches=0.0 )
        plt.clf()
        plt.close()

        ##save also data in the folder as csv
        txt_name = sample_name + "_1D.csv"
        np.savetxt(sample_folder+txt_name, data, delimiter=",")

        label_name = sample_name + "_labels.csv"
        np.savetxt(sample_folder+label_name, labels, delimiter=",")

        print("Saved "+str(i)+" image: ", img_name)




if __name__ == "__main__":

    #parse --get only tensor flag
    parser = argparse.ArgumentParser(description='Generate 2D dataset from 1D dataset')
    parser.add_argument('--save_images', action='store_true', help='Save dataset as images and readable files')
    parser.add_argument('--smaller_tensors', action='store_true', help='Save dataset as tensors')

    args = parser.parse_args()

    input_path = "dataset_copernicus2/"
    folder_name = "2D_Dataset_copernicus/"

    size = (224, 224)
    
    if args.smaller_tensors:
        size = (32, 32)


    if os.path.exists(folder_name):
        #ask if user wants to delete the folder
        print("Folder already exists: ", folder_name)
        print("Do you want to delete it? (y/n)")
        answer = input()
        if answer == "n":
            sys.exit()
        shutil.rmtree(folder_name)
        print("Deleted folder: ", folder_name)
        
    else:
        os.makedirs(folder_name)
        print("Created folder: ", folder_name)

    
    generate_cwt_from_1Ddataset(data_path = input_path, output_path= folder_name, size= size, save_dataset_as_images= args.save_images)






    # sample = dataset[0][0].numpy()

    # if print_wave_samples:

    #     wave_list = pywt.wavelist()

    #     for wave in wave_list:

    #         try:

    #             coef, freqs=pywt.cwt(sample,np.arange(1,129),wave)

    #             plt.matshow(coef) 
    #             plt.savefig("morl", bbox_inches='tight',pad_inches=0.0 )
    #             plt.clf()
    #             plt.close()

    #             print("Done with wavelet: ", wave)






# dataset = Dataset_1D_copernicus(
#     csv_file='filtered_datasets_copericus_csv/1Ddataset_45.60_13.54/1_45.60_13.54.csv',
#     window_size=24*7,
#     output_size=24,
# )


# sample = dataset[0][0].numpy()


# print_wave_samples = False

# if print_wave_samples:
    
#     wave_list = pywt.wavelist()

#     for wave in wave_list:

#         try:

#             coef, freqs=pywt.cwt(sample,np.arange(1,129),wave)

#             plt.matshow(coef) 
#             plt.savefig(folder_name+"test_cwt"+str(wave), bbox_inches='tight',pad_inches=0.0 )
#             plt.clf()
#             plt.close()

#             print("Done with wavelet: ", wave)

#         except:

#             print("Error with wavelet: ", wave)
                
#             continue
