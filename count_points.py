import os
import glob
import laspy
from tqdm import tqdm

def count_total_laz_points(folder_path, file_extension="*.laz"):
    """
    Scans a folder for .laz (or .las) files and reads the header 
    to instantly get the point count without loading the data.
    """
    # Find all matching files in the folder and subfolders
    search_path = os.path.join(folder_path, "**", file_extension)
    file_list = glob.glob(search_path, recursive=True)
    
    if not file_list:
        print(f"No '{file_extension}' files found in: {folder_path}")
        return

    print(f"Found {len(file_list)} files. Reading headers...")
    
    total_points = 0
    
    # Loop through the files with a progress bar
    for file_path in tqdm(file_list, desc="Counting LAZ Points"):
        try:
            # laspy.open() only reads the header, making it lightning fast!
            with laspy.open(file_path) as f:
                total_points += f.header.point_count
                
        except Exception as e:
            print(f"\nError reading {file_path}: {e}")

    # Format the final number with commas
    print("\n" + "="*40)
    print(f"GRAND TOTAL: {total_points:,} points")
    print("="*40)

if __name__ == "__main__":
    # Point this to your raw LAZ directory
    TARGET_FOLDER = "/mnt/d/PointCloudsFiles/Unclassified_Data"
    
    count_total_laz_points(TARGET_FOLDER, "*.laz")
    
    # Tip: If you also have uncompressed .las files, you can just call it again:
    # count_total_laz_points(TARGET_FOLDER, "*.las")
