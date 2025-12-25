#!/usr/bin/env python3
"""
Convert seismic train_list.txt from H5 format to BIPED image format.
Changes: ci38591583.h5 → SEISMIC_IMG/ci38591583.png SEISMIC_IMG/ci38591583.png
"""

import os
from pathlib import Path
import argparse

def convert_list_file(input_list, output_list=None, img_dir='../SEISMIC_IMG'):
    """
    Convert H5 list to BIPED format (image label pairs).
    
    Args:
        input_list: Path to input list file (contains .h5 filenames)
        output_list: Path to output list file (default: same as input)
        img_dir: Directory containing PNG images relative to input_list location (default: ../SEISMIC_IMG)
    """
    
    if output_list is None:
        output_list = input_list
    
    print(f"Reading: {input_list}")
    
    with open(input_list, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(lines)} entries\n")
    
    # Convert each line
    new_lines = []
    for line in lines:
        # If line already has two parts (is already in BIPED format), skip
        parts = line.split()
        if len(parts) == 2:
            new_lines.append(line)
            print(f"Already formatted: {line}")
        else:
            # Remove .h5 extension and add .png
            base_name = Path(line).stem
            png_path = f"{img_dir}/{base_name}.png"
            
            # Format: image_path label_path (same image for both)
            new_line = f"{png_path} {png_path}"
            new_lines.append(new_line)
            print(f"{line:30} → {new_line}")
    
    # Write output
    print(f"\nWriting: {output_list}")
    with open(output_list, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')
    
    print(f"✓ Converted {len(new_lines)} entries")

def main():
    parser = argparse.ArgumentParser(description='Convert seismic list to BIPED format')
    parser.add_argument('input_list', type=str, 
                       help='Path to input list file (train_list.txt or test_list.txt)')
    parser.add_argument('--output_list', type=str, default=None,
                       help='Path to output list file (default: overwrite input)')
    parser.add_argument('--img_dir', type=str, default='../SEISMIC_IMG',
                       help='Directory containing PNG images relative to list file (default: ../SEISMIC_IMG)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_list):
        print(f"Error: {args.input_list} not found")
        return
    
    # If no output specified, overwrite input
    output = args.output_list if args.output_list else args.input_list
    convert_list_file(args.input_list, output, args.img_dir)

if __name__ == '__main__':
    main()