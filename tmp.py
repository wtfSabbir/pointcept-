import numpy as np
from pathlib import Path

sample_dir = "/mnt/d/PointCloudsFiles/Pointcept/Refinement_PTv3/DataV2/train"
sample = sorted(Path(sample_dir).iterdir())[0]

v1 = np.load(sample / "v1_pred.npy")
print(f"v1_pred.npy shape:         {v1.shape}")
print(f"v1_pred.npy dtype:         {v1.dtype}")
print(f"v1_pred.npy min:           {v1.min():.6f}")
print(f"v1_pred.npy max:           {v1.max():.6f}")
print(f"v1_pred.npy mean:          {v1.mean():.6f}")
print(f"v1_pred.npy unique values: {np.unique(v1)}")