import os
import glob

IMG_DIR = "data/SEISMIC_IMG"
OUTPUT_FILE = "seismic_list.txt"

png_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))

with open(OUTPUT_FILE, "w") as f:
    for p in png_files:
        f.write(f"{p}\n")

print(f"[INFO] Created {OUTPUT_FILE} with {len(png_files)} entries.")
