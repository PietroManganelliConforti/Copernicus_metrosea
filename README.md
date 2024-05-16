# Copernicus_metrosee

MetroSee code repository for the implementation of copernicus dataset values forecast from volumetric temperature trends transformed through CWT and analyzed in parallel from 16 neural network simultaneously.


## HOW TO RUN:

From the repository folder, to generate the 1D dataset, run:

``` docker run -it -v $(pwd):/app --gpus all piemmec/copernicus sh -c "cd app && python3 data_to_csv.py && python3 filter_dataset.py" ```

From here, to generate the 2D dataset run:

``` docker run -v $(pwd):/app --gpus all piemmec/copernicus sh -c "cd app && python3 generate_2D_dataset.py --only_tensors" ```

to run python code, use:

``` docker run -it -v $(pwd):/app --gpus all piemmec/copernicus sh -c "cd app && python3 main.py" ```
