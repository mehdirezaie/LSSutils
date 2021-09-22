"""
    Script to prepare log-normal mock data tables for regression
"""
import sys
import os
import fitsio as ft
import numpy as np
import healpy as hp
import lssutils.utils as utils


# read mock
mock_path = sys.argv[1]
output_path = sys.argv[2]
region = sys.argv[3] #"bmzls"
print(f"input mock: {mock_path}")
print(f"input region: {region}")
print(f"output table: {output_path}")
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    print("output dir does not exist")
    os.makedirs(output_dir)
mock = hp.read_map(mock_path, dtype=np.float32, verbose=False)
print(f"mock nside: {np.sqrt(mock.size/12)}")

# read imaging tables
root_dir = "/fs/ess/PHS0336/data/"
table_path = f"{root_dir}rongpu/imaging_sys/tables/nlrg_features_{region}_256.fits"
table = ft.read(table_path)
print(f"input table: {table.dtype}")

# do the main thing
label = mock[table['hpix']]
table['label'] = label*1.0
table['fracgood'] = 1.0

# save
print(table)
ft.write(output_path, table)
