import os
from utils.io import extract_data
from utils.plot import plot_seismogram

print("Starting seismogram conversion...")

files = [
    "../quakeflow_das/data/ridgecrest_north/ci37280444.h5",
    "../quakeflow_das/data/ridgecrest_north/ci37280604.h5"
]

# generate seismogram images
for filepath in files:
    data, dt = extract_data(filepath)
    filename = os.path.splitext(os.path.basename(filepath))[0]
    outpath = f"results/seismograms/{filename}.png"
    plot_seismogram(data, dt, outpath)
    print(f"Saved: {outpath}")