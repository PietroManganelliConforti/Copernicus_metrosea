import matplotlib.pyplot as plt
import getpass
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import copernicusmarine
import os
import shutil
from tqdm import tqdm


## 1. Create the dataset folder dataset_copernicus

## 2. Download the dataset from Copernicus Marine Data Store CLI, 
## using copernicusmarine login credentials
## and copernicusmarine CLI commands

# #Coordinates 2D:
#   * latitude   (latitude) floatn32 16B 45.6 45.65 45.69 45.73
#   * longitude  (longitude) float32 20B 13.5 13.54 13.58 13.62 13.67


# #Coordinates 3D:
#   * depth      (depth) float32 96B 1.018 3.166 5.465 7.92 ... 84.74 91.2 97.93
#   * latitude   (latitude) float32 16B 45.6 45.65 45.69 45.73
#   * longitude  (longitude) float32 20B 13.5 13.54 13.58 13.62 13.67
#   * time       (time) datetime64[ns] 35kB 2023-11-13 ... 2024-05-12T23:00:00


## 3. Convert it with the following code:


dataset_folder = 'dataset_cop2/'
new_dataset_folder = "dataset_cop2_csv/"


# Funzione per convertire i file .nc in .csv
def convert_to_csv(src_file, dest_file):

    DS_sample = xr.open_dataset(src_file)
    DS_sample.to_dataframe().to_csv(dest_file)

# Percorre tutte le cartelle e sottocartelle
for root, dirs, files in os.walk(dataset_folder):
    # Crea la struttura delle cartelle nel nuovo dataset_folder
    new_root = root.replace(dataset_folder, new_dataset_folder)
    os.makedirs(new_root, exist_ok=True)
    print(root, new_root, dirs, files)
    # Copia i file mantenendo la stessa struttura di cartelle e convertendo i .nc in .csv
    for file in tqdm(files):
        print(file)
        if file.endswith('.nc'):
            src_file = os.path.join(root, file)
            dest_file = os.path.join(new_root, file.replace('.nc', '.csv'))
            convert_to_csv(src_file, dest_file)
        else:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(new_root, file)
            shutil.copy(src_file, dest_file)
