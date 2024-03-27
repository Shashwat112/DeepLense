from data import path, download_data, url, exp_name
from data.utils import pair_data, unpaired_data
import numpy as np

global cont_data
if download_data:
    # Code for downloading/web scraping the data from the given url
    pass
else:
    if exp_name == 'super_resolution':
        cont_data: np.ndarray = pair_data(path)
    if exp_name == 'diffusion':
        cont_data: np.ndarray = unpaired_data(path)
