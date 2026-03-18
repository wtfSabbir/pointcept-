import os
import numpy as np
from tqdm import tqdm

data_root = "/mnt/d/PointCloudsFiles/ptv3/data/npy/train"
files = [f for f in os.listdir(data_root) if f.endswith('.npy')]

print(f"Checking {len(files)} files for coordinate issues...")

for f in tqdm(files):
    data = np.load(os.path.join(data_root, f))
    
    # Check shape - are there definitely 5 columns?
    if data.shape[1] < 5:
        print(f"\nERROR: File {f} only has {data.shape[1]} columns!")
        continue

    coords = data[:, :3]
    
    # Check for extreme values
    c_max = np.abs(coords).max()
    if c_max > 100000: # If any point is > 100km from origin
        print(f"\nWARNING: Extreme coordinates in {f}: Max Abs = {c_max:.2f}")

    # Check for NaN in coords
    if np.isnan(coords).any():
        print(f"\nCRITICAL: NaN found in coordinates of {f}")