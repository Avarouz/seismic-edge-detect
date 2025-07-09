
import numpy as np
import matplotlib.pyplot as plt
import os

'''
Plots seismic waveform image and saves it

'''
def plot_seismogram(data, dt, save_path):

    n_channels, n_samples = data.shape
    t_axis = np.arange(n_samples) * dt

    plt.figure(figsize=(12, 6))
    plt.imshow(data, aspect='auto', cmap='seismic', extent=[t_axis[0], t_axis[-1], n_channels, 0])
    plt.colorbar(label='Microstrain/s')
    plt.xlabel('Time (s)')
    plt.ylabel('Channel')
    plt.title('Seismogram')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()