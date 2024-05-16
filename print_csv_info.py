import pandas as pd

# Load the dataset
df = pd.read_csv("dataset_cop2_csv/med-cmcc-tem-rean-d_bottomT-thetao_13.54E-13.67E_45.60N-45.73N_1.02-97.93m_1987-01-01-2022-07-31.csv")

# Group data by depth and check if all thetao values exist for each depth
depths_with_all_thetao = []
for depth, group in df.groupby('depth'):
    if len(group['thetao'].dropna()) == len(group):
        depths_with_all_thetao.append(depth)

# Print depths where all thetao values exist
if depths_with_all_thetao:
    print("Depths where all thetao values exist:")
    print(depths_with_all_thetao)
else:
    print("No depths found where all thetao values exist.")