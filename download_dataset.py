#from IPython.display import IFrame
#%matplotlib inline
import matplotlib.pyplot as plt
import pydap
import getpass
import xarray as xr
import panel.widgets as pnw
import panel as pn
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pydap.client import open_url
from pydap.cas.get_cookies import setup_session
import getpass
from pathlib import Path
from platform import system
from os.path import exists

# To avoid warning messages
import warnings
warnings.filterwarnings('ignore')

#datasetID = 'cmems_obs-ins_med_phybgcwav_mynrt_na_irr_202211--ext--monthly'
datasetID = 'dataset-duacs-nrt-global-merged-allsat-phy-l4'
USERNAME = 'prusso'
PASSWORD = 'wd@KVpxX6pF3ped'


HOME = Path.home()
netrc_file = HOME / "_netrc" if system() == "Windows" else HOME / ".netrc"
dodsrc_file = HOME / ".dodsrc"
cookies_file = HOME / ".cookies"
OPeNDAP_SERVERS = ["my.cmems-du.eu", "nrt.cmems-du.eu"]

if not exists(netrc_file):
    username = USERNAME
    password = PASSWORD

    # Create netrc file
    with open(netrc_file, "a") as file:
        for server in OPeNDAP_SERVERS:
            file.write(f"machine {server}\nlogin {username}\npassword {password}\n\n")
        
if not exists(dodsrc_file):
    # Create dodsrc file
    with open(dodsrc_file, "a") as file:
        file.write(f"HTTP.NETRC={netrc_file}\nHTTP.COOKIEJAR={cookies_file}")

## OPeNDAP connection
def copernicusmarine_datastore(dataset, username, password):
    from pydap.client import open_url
    from pydap.cas.get_cookies import setup_session
    cas_url = 'https://cmems-cas.cls.fr/cas/login'
    session = setup_session(cas_url, username, password)
    try:
        session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
    except:
        print("Bad credentials. Please try again.")
    database = ['my', 'nrt']
    url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
    try:
        data_store = xr.backends.PydapDataStore(open_url(url, session=session))
    except:
        url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
        data_store = xr.backends.PydapDataStore(open_url(url, session=session))
    return data_store

dataset_connection = copernicusmarine_datastore(datasetID, USERNAME, PASSWORD)