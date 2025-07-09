import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DATA_DIR = Path("/home/joe/research/seismic-edge-detect/ridgecrest_north")
SAVE_PLOT = False

def normalize_rows(data, clip_val=1.0):
    normalized = data / (np.max(np.abs(data), axis=1, keepdims=True) + 1e-6)
    return np.clip(normalized, -clip_val, clip_val)


def plot_arrival_image(data, dt_s, event_time_index, event_id):
    num_channels, num_samples = data.shape
    time = np.arange(num_samples) * dt_s

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data.T, aspect='auto', cmap='seismic',
                   extent=[0, num_channels, time[-1], time[0]],
                   vmin=-1, vmax=1, origin='upper') 
    
    event_time = event_time_index * dt_s
    ax.axhline(event_time, color='yellow', linestyle='--', linewidth=1.5, label='Event Time')

    ax.set_title(f"Seismogram - {event_id}")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Channel")
    plt.colorbar(im, ax=ax, label="Normalized Strain Rate")

    if SAVE_PLOT:
        plt.savefig(f"{event_id}_arrival_image_flipped.png")
    plt.legend()
    plt.tight_layout()
    plt.show()

def process_h5_file(filepath):
    with h5py.File(filepath, "r") as f:
        raw = f["data"][:]  # (1150, 12000)
        dt_s = float(f["data"].attrs["dt_s"])
        event_time_index = int(f["data"].attrs["event_time_index"])
        event_id = f["data"].attrs["event_id"]

        processed = normalize_rows(raw, clip_val=1.0)
        plot_arrival_image(processed, dt_s, event_time_index, event_id)


all_h5_files = sorted(DATA_DIR.glob("*.h5"))[:5]  # adjust as needed
for file in all_h5_files:
    process_h5_file(file)
