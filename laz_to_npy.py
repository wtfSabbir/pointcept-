import numpy as np
import laspy
import os
from tqdm import tqdm

def convert_laz_to_npy(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all LAZ files
    files = [f for f in os.listdir(input_folder) if f.endswith('.laz')]
    print(f"Found {len(files)} files. Starting conversion...")

    for filename in tqdm(files):
        # 1. Load the LAZ file
        las = laspy.read(os.path.join(input_folder, filename))
        
        # 2. Extract Raw Data (NO DOWNSAMPLING)
        # We use 'scaled_x' etc. to get the real-world floating point coordinates
        points = np.vstack((
            las.x, 
            las.y, 
            las.z, 
            las.intensity, 
            las.classification
        )).transpose()

        # 3. Save as NPY
        output_name = filename.replace('.laz', '.npy')
        np.save(os.path.join(output_folder, output_name), points.astype(np.float32))

# --- RUN THE CONVERSION ---
# Replace these with your actual paths
INPUT_DIR = "/mnt/d/PointCloudsFiles/Lidar_Annotation/input"
OUTPUT_DIR = "/mnt/d/PointCloudsFiles/Lidar_Annotation/input"

convert_laz_to_npy(INPUT_DIR, OUTPUT_DIR)