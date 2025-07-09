import os

from utils.io import extract_data
from utils.plot import plot_seismogram
print(os.getcwd())

print("Starting seismogram conversion...")

import h5py
f = h5py.File('../quakeflow_das/data/ridgecrest_north/ci37280444.h5', 'r')
print(list(f.keys()))
f.close()


files = [
    "../quakeflow_das/data/ridgecrest_north/ci37280444.h5",
    "../quakeflow_das/data/ridgecrest_north/ci37280604.h5"
]

# generate seismogram images
for filepath in files:
    print("Current working directory:", os.getcwd())
    print("Trying to open:", filepath)
    print("File exists:", os.path.exists(filepath))

    data, dt = extract_data(filepath)
    filename = os.path.splitext(os.path.basename(filepath))[0]
    outpath = f"results/seismograms/{filename}.png"
    plot_seismogram(data, dt, outpath)
    print(f"Saved: {outpath}")