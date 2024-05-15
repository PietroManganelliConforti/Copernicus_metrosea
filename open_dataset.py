import matplotlib.pyplot as plt
import pydap
import getpass
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import os

def iglob(dir='./',ext='.nc'):
    pys = []
    for file in glob.iglob(os.path.join(dir,'**/*'), recursive=True):
        if file.endswith(ext):
            pys.append(file)
    return pys


file_list = iglob('./dataset/')
import pdb; pdb.set_trace()


