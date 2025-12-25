#!/usr/bin/env python3
"""
Convert seismic H5 files to PNG images for TEED edge detection.
"""

import os
import h5py
import numpy as np
import cv2
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
import matplotlib.cm as cm
import argparse

def bandpass(data, fs, lowcut=2, highcut=10, order=4):
    """Apply bandpass filter to seismic data."""
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=1)

def load_seismic_h5(h5_path):
    """Load seismic H5 file."""
    with h5py.File(h5_path, "r") as fp:
        ds = fp["data"]
        data = ds[...]  # (channels, time)
        dt = float(ds.attrs.get("dt_s", 0.005))
        dx = float(ds.attrs.get("dx_m", 1.0))
    return data, dt, dx

def h5_to_png(h5_path, output_dir, height=1024, width=1024, colormap='seismic'):
    """
    Convert H5 file to PNG image.
    
    Args:
        h5_path: Path to H5 file
        output_dir: Directory to save PNG
        height: Output image height
        width: Output image width
        colormap: Matplotlib colormap name
    
    Returns:
        Path to output PNG file
    """
    try:
        # Load H5
        data, dt, dx = load_seismic_h5(h5_path)
        print(f"Loaded {h5_path}: shape={data.shape}, dt={dt}, dx={dx}")
        
        # Bandpass filter
        fs = 1.0 / dt
        data = bandpass(data, fs, lowcut=2, highcut=10)
        
        # Robust clipping at 99th percentile
        clip_val = np.percentile(np.abs(data), 99)
        data = np.clip(data, -clip_val, clip_val)
        
        # Normalize to [-1, 1]
        data = data / (clip_val + 1e-6)
        
        # Resize (channels → height, time → width)
        data = cv2.resize(data, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Map to [0, 1] and apply colormap
        data = (data + 1) / 2
        cmap = cm.get_cmap(colormap)
        rgba = cmap(data)
        rgb = rgba[..., :3]
        
        # Convert to uint8
        rgb_image = (rgb * 255).astype(np.uint8)
        
        # Save PNG
        base_name = Path(h5_path).stem
        output_path = os.path.join(output_dir, f"{base_name}.png")
        cv2.imwrite(output_path, rgb_image)
        
        print(f"  → Saved: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Convert seismic H5 files to PNG')
    parser.add_argument('input_dir', type=str, help='Directory containing H5 files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: SEISMIC_IMG folder)')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--colormap', type=str, default='seismic',
                       help='Matplotlib colormap (seismic, viridis, hot, cool, etc.)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input_dir), 'SEISMIC_IMG')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Converting H5 files from: {args.input_dir}")
    print(f"Saving PNG to: {args.output_dir}")
    print(f"Image size: {args.height}x{args.width}, Colormap: {args.colormap}\n")
    
    # Find all H5 files
    h5_files = sorted(Path(args.input_dir).glob('*.h5'))
    print(f"Found {len(h5_files)} H5 files\n")
    
    if not h5_files:
        print("No H5 files found!")
        return
    
    # Convert each H5 to PNG
    successful = 0
    for h5_file in h5_files:
        output_path = h5_to_png(str(h5_file), args.output_dir, 
                               args.height, args.width, args.colormap)
        if output_path:
            successful += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion complete: {successful}/{len(h5_files)} files converted")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()