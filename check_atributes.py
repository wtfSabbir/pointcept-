import laspy

INPUT_FILE = "/mnt/d/PointCloudsFiles/Lidar_Annotation/input/39346_56722_chunked_sorted.copc_chunked_sorted.laz"

def check_attributes():
    # Open the file in read mode without loading the heavy point data into memory
    with laspy.open(INPUT_FILE) as f:
        print(f"--- File Metadata for {INPUT_FILE} ---")
        print(f"Point Format ID: {f.header.point_format.id}")
        print(f"Total Points: {f.header.point_count}")
        
        print("\n--- Available Attributes (Dimensions) ---")
        # Fetch and print the list of all available dimension names
        dimensions = list(f.header.point_format.dimension_names)
        for dim in dimensions:
            print(f"- {dim}")

if __name__ == "__main__":
    check_attributes()