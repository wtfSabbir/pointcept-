import torch, numpy as np, os, laspy
from tqdm import tqdm
from scipy.spatial import cKDTree
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.models.utils.structure import Point

def main():
    config_path = "pointcept/configs/laz_static/lidar_config.py"
    checkpoint_path = "exp/laz_static/lidar_ptv3_run1/model/model_last.pth"
    input_folder = "/mnt/d/PointCloudsFiles/Lidar_Annotation/input" 
    output_folder = "/mnt/d/PointCloudsFiles/Lidar_Annotation/output"
    os.makedirs(output_folder, exist_ok=True)

    cfg = Config.fromfile(config_path)
    cfg.model.backbone.enable_flash = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint)))
    model.eval()

    # The model "thinks" in 5cm voxels
    test_transform = Compose([
        dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "strength"))
    ])

    files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    
    for filename in tqdm(files):
        raw_data = np.load(os.path.join(input_folder, filename))
        orig_xyz, orig_intensity = raw_data[:, :3], raw_data[:, 3]
        mean_xyz = orig_xyz.mean(axis=0)

        # 1. Model inference on 5cm summary
        data_dict = test_transform(dict(coord=orig_xyz-mean_xyz, strength=orig_intensity.reshape(-1,1), segment=np.zeros(len(orig_xyz), dtype=np.int64)))
        for k in data_dict.keys(): 
            if isinstance(data_dict[k], torch.Tensor): data_dict[k] = data_dict[k].to(device)
        
        n_s = data_dict["coord"].shape[0]
        data_dict["offset"], data_dict["batch"] = torch.tensor([n_s], device=device).int(), torch.zeros(n_s, device=device).int()
        
        with torch.no_grad():
            output = model(Point(data_dict))
            sampled_labels = (output["seg_logits"] if isinstance(output, dict) else output).max(1)[1].cpu().numpy()

        # 2. THE RESTORATION: Match model results to ORIGINAL high-density points
        # This is what makes it SOLID
        tree = cKDTree(data_dict["coord"].cpu().numpy())
        _, idx = tree.query(orig_xyz - mean_xyz)
        full_labels = sampled_labels[idx]

        # 3. SAVE AS SOLID LAZ (No points missing)
        las = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
        las.x, las.y, las.z = orig_xyz[:, 0], orig_xyz[:, 1], orig_xyz[:, 2] # ORIGINAL POINTS
        las.intensity = orig_intensity.astype(np.uint16)
        las.classification = full_labels.astype(np.uint8)
        las.write(os.path.join(output_folder, filename.replace('.npy', '.laz')))

if __name__ == "__main__":
    main()