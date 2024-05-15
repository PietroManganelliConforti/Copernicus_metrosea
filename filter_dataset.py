import pandas as pd
import os 

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset_cop2_csv/med-cmcc-tem-rean-d_bottomT-thetao_13.54E-13.67E_45.60N-45.73N_1.02-97.93m_1987-01-01-2022-07-31.csv')

# Filter the DataFrame
target_depths = [1.0182366, 3.1657474, 5.4649634]

filtered_df = df[  (df['depth'].isin(target_depths))]

# drop the 5th column
filtered_df = filtered_df.drop(filtered_df.columns[4], axis=1)

filtered_data_path = 'filtered_dataset_cop2/'

if not os.path.exists(filtered_data_path):
    os.makedirs(filtered_data_path)
    print("Created folder: ", filtered_data_path)

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv(filtered_data_path+'filtered_dataset_cop2.csv', index=False)



unique_latitudes = filtered_df['latitude'].unique() # [45.604168 45.645832 45.6875 45.729168]
unique_longitudes = filtered_df['longitude'].unique() # [13.541667 13.583333 13.625 13.666667]


# Generate 16 datasets
for lat in unique_latitudes:
    for lon in unique_longitudes:
        for dep in target_depths:
            
            # Save the filtered DataFrame to a new CSV file
            folder = f"1Ddataset_{str(lat)[:5]}_{str(lon)[:5]}/"

            filename = f"{int(dep)}_{str(lat)[:5]}_{str(lon)[:5]}.csv"

            if not os.path.exists("dataset_copernicus2/"+folder):
                os.makedirs("dataset_copernicus2/"+folder)
                print("created folder: ",folder)


            filtered_df_single = filtered_df[(filtered_df['latitude'] == lat) &
                                             (filtered_df['longitude'] == lon) &
                                                (filtered_df['depth'] == dep)]
            #save csv in folder + filename

            
            filtered_df_single.to_csv("dataset_copernicus2/"+folder+filename, index=False)

            print("Saved: ", "dataset_copernicus2/"+folder+filename)