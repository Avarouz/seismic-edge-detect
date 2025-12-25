#!/usr/bin/env python3
"""
Generate seismic_list.txt and seismic_train_list.txt for SEISMIC dataset.
"""
import os
from pathlib import Path

# Set your SEISMIC data directory here
SEISMIC_DIR = Path.home() / "seismic-edge-detect" / "data" / "SEISMIC"

def generate_seismic_lists():
    """Generate train and test list files for seismic H5 data."""
    
    # Check if directory exists
    if not SEISMIC_DIR.exists():
        print(f"ERROR: Directory not found: {SEISMIC_DIR}")
        return False
    
    # Find all H5 files
    h5_files = sorted(SEISMIC_DIR.glob("*.h5"))
    
    if not h5_files:
        print(f"ERROR: No H5 files found in {SEISMIC_DIR}")
        return False
    
    print(f"Found {len(h5_files)} H5 files:")
    for f in h5_files:
        print(f"  - {f.name}")
    
    # Split into train and test (80/20 split)
    num_train = int(len(h5_files) * 0.8)
    train_files = h5_files[:num_train]
    test_files = h5_files[num_train:]
    
    print(f"\nSplit: {len(train_files)} train, {len(test_files)} test")
    
    # Write train list
    train_list_path = SEISMIC_DIR / "seismic_train_list.txt"
    with open(train_list_path, 'w') as f:
        for file in train_files:
            f.write(f"{file.name}\n")
    print(f"✓ Created: {train_list_path}")
    
    test_list_path = SEISMIC_DIR / "seismic_list.txt"
    with open(test_list_path, 'w') as f:
        for file in test_files:
            f.write(f"{file.name}\n")
    print(f"✓ Created: {test_list_path}")
    
    all_list_path = SEISMIC_DIR / "seismic_all_list.txt"
    with open(all_list_path, 'w') as f:
        for file in h5_files:
            f.write(f"{file.name}\n")
    print(f"✓ Created: {all_list_path}")
    
    print("\nYou can now use:")
    print(f"  --train_list {train_list_path.name}")
    print(f"  --test_list {test_list_path.name}")
    
    return True

if __name__ == "__main__":
    generate_seismic_lists()