# Copernicus_metrosee

MetroSee code repository for the implementation of copernicus dataset values forecast from volumetric temperature trends transformed through CWT and analyzed in parallel from 16 neural network simultaneously.

## Details

- This project uses a docker image called piemmec/copernicus, loaded on dockerhub
- Dataset: "https://data.marine.copernicus.eu/product/MEDSEA_ANALYSISFORECAST_PHY_006_013/"



## 1D dataset generation

From the repository folder, run:

``` docker run -it -v $(pwd):/app --gpus all piemmec/copernicus sh -c "cd app && python3 data_to_csv.py && python3 filter_dataset.py" ```

## 2D dataset generation

``` docker run -v $(pwd):/app --gpus all piemmec/copernicus sh -c "cd app && python3 generate_2D_dataset.py --only_tensors" ```

## Training and evaluation

``` docker run -it -v $(pwd):/app --gpus all piemmec/copernicus sh -c "cd app && python3 main.py" ```
