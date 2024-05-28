# Copernicus_metrosea

## Overview

MetroSea 2024 research project implementation of copernicus values forecast from volumetric temperature trends transformed through CWT and analyzed simultaneously in parallel from 16 neural network. Temporal trends evaluations from each network are combined and analyzed through an ensemble learning approach.

## Details

- This project uses a docker image called piemmec/copernicus, loaded on dockerhub
- Dataset: "https://data.marine.copernicus.eu/product/MEDSEA_ANALYSISFORECAST_PHY_006_013/"



## 1D dataset generation

From the repository folder, run:

``` docker run -it -v $(pwd):/app --gpus all piemmec/copernicus sh -c "cd app && python3 data_to_csv.py && python3 filter_dataset.py" ```

## 1D dataset exploration

``` docker run -it -v $(pwd):/app --gpus all piemmec/copernicus sh -c "cd app && python3 dataset1D.py" ```

## 2D dataset generation

``` docker run -v $(pwd):/app --gpus all piemmec/copernicus sh -c "cd app && python3 generate_2D_dataset.py" ```

To save the dataset in an explorable way, and not only the torch tensors, run the last command with the "--save_images" flag for the generate_2D_dataset.py script

## Training and evaluation

``` docker run -it -v $(pwd):/app --gpus all piemmec/copernicus sh -c "cd app && python3 main.py" ```
