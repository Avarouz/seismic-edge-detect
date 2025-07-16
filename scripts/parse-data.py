import os
import torch
import numpy as np
import h5py
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd

if not os.path.exists("data"):
    os.mkdir("data")

event_id = "ci38595978"
if not os.path.exists(f'data/{event_id}.h5'):
    os.system(f"wget https://huggingface.co/datasets/AI4EPS/quakeflow_das/resolve/main/data/ridgecrest_north/{event_id}.h5 -P data > log.txt 2>&1")

normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)
# h5_files = glob("data/*.h5")
# for file in h5_files:
file = f"data/{event_id}.h5"
with h5py.File(file, "r") as fp:
    ds = fp["data"]
    data = ds[...]
    dt = ds.attrs["dt_s"]
    dx = ds.attrs["dx_m"]
    nx, nt = data.shape
    x = np.arange(nx) * dx
    t = np.arange(nt) * dt
        
plt.figure()
plt.imshow(normalize(data).T, cmap="seismic", vmin=-1, vmax=1, aspect="auto", extent=[x[0], x[-1], t[-1], t[0]], interpolation="none")
plt.xlabel("Distance (m)")
plt.ylabel("Time (s)")
plt.show()

# LOCAL_DIR = "./ridgecrest_north/"
# NUM_FILES = 2

# print("Looking for local H5 files")
# print(f"Searching in: {os.path.abspath(LOCAL_DIR)}")

# if not os.path.exists(LOCAL_DIR):
#     print(f"Directory {LOCAL_DIR} does not exist.")
#     print(f"Current working directory: {os.getcwd()}")
#     exit()

# h5_files = glob.glob(os.path.join(LOCAL_DIR, "*.h5"))
# h5_files = sorted(h5_files)

# if not h5_files: # error handling
#     print("No H5 files found")
#     exit()


# # display what we found
# print(f"H5 Files found, showing first {NUM_FILES} files")

# for i, f in enumerate(h5_files[:NUM_FILES]):
#     print(f"{i+1} {os.path.basename(f)}")
    

# files_to_process = h5_files[:NUM_FILES]
# print(f"\n Processing {len(files_to_process)} files")


# # inspection of H5 files
# for i, file_path in enumerate(files_to_process):
#     print(f"Inspecting files {i+1}: {os.path.basename(file_path)}")

#     try:
#         with h5py.File(file_path, "r") as f:
#             print(f"Root level keys: {list(f.keys())}")

#             if f.attrs:  # Check root attributes first
#                 print(f"Root attributes: {dict(f.attrs)}")

#             def explore(name, obj):
#                 indent = "  " * (name.count('/') + 1)
                
#                 if isinstance(obj, h5py.Dataset):
#                     print(f"{indent}{name}")  # Fixed: was "name" instead of {name}
#                     print(f"{indent} Shape: {obj.shape}")
#                     print(f"{indent} dType: {obj.dtype}")
#                     print(f"{indent} Size: {obj.size} elements")

#                     # add sample attempt later...

#                     if obj.attrs:
#                         print(f"{indent} Attributes: {dict(obj.attrs)}")

#             f.visititems(explore)

#     except Exception as e:
#         print(f"Could not open {file_path}: {e}")
#         print(f"Error type: {type(e).__name__}")

#         if os.path.exists(file_path):
#             print(f"File exists, size: {os.path.getsize(file_path)} bytes")
#         else:
#             print(f"File does not exist")

# print(f"\n Inspection complete! :)")