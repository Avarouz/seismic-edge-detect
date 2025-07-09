import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt


DATA_DIR = Path("/home/joe/research/seismic-edge-detect/ridgecrest_north") # dawg i need to change this
TEED_DATA_DIR = Path("TEED/data")
TEED_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
SAVE_PLOT = True

def normalize_rows(data, clip_val=1.0):
    normalized = data / (np.max(np.abs(data), axis=1, keepdims=True) + 1e-6)
    return np.clip(normalized, -clip_val, clip_val)

def bandpass_filter(data, fs, low=1.0, high=10.0, order=4):
    b, a = butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data, axis=1)


def plot_arrival_image(data, dt_s, event_time_index, event_id):
    num_channels, num_samples = data.shape
    time = np.arange(num_samples) * dt_s
    event_time = event_time_index * dt_s

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data, aspect='auto', cmap='seismic',
                   extent=[time[0], time[-1], 0, num_channels],
                   vmin=-0.3, vmax=0.3, origin='upper')
    
    
    
    ax.axvline(event_time, color='yellow', linestyle='--', linewidth=1.5, label='Event Time')

    ax.set_title(f"Seismogram - {event_id}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")

    plt.colorbar(im, ax=ax, label="Normalized Strain Rate")
    plt.legend()
    plt.tight_layout()

    if SAVE_PLOT:
        save_path = TEED_DATA_DIR / f"seismogram_{event_id}.png"

        if save_path.exists():
            print(f"File {save_path} already exists. Skipping save.")
        else:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")

    plt.show()
    plt.close(fig)

def process_h5_file(filepath):

    with h5py.File(filepath, "r") as f:
        raw = f["data"][:]  # (1150, 12000), channels and samples
        dt_s = float(f["data"].attrs["dt_s"])
        event_time_index = int(f["data"].attrs["event_time_index"])
        event_id = f["data"].attrs["event_id"]

        fs = 1 / dt_s # sample f (Hz)
        filtered = bandpass_filter(raw, fs)

        processed = normalize_rows(filtered, clip_val=1.0)
        plot_arrival_image(processed, dt_s, event_time_index, event_id)

        print(f"{filepath.name} shape: {raw.shape}")



all_h5_files = sorted(DATA_DIR.glob("*.h5"))[:10]  # adjust as needed
for file in all_h5_files:
    process_h5_file(file)
