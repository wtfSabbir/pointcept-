"""
pointcept/datasets/lidar_dataset2.py

v2 dataset - same as lidar_dataset.py but reads v1_pred.npy as second feature.
Input features: [intensity, v1_pred] -> color shape (N, 2)
in_channels in config must be 5 (XYZ=3 + intensity=1 + v1_pred=1)

Place in: pointcept/datasets/lidar_dataset2.py
Add to pointcept/datasets/__init__.py:
    from .lidar_dataset2 import *
"""

import os
import numpy as np
from torch.utils.data import Dataset
from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class LidarDatasetV2(Dataset):

    class_names = [
        "Unclassified", "Ground", "Pole", "Sign", "Bollard",
        "Trunk", "Vegetation", "Building", "Fence", "Gate",
    ]

    def __init__(self, split="train", data_root="data/lidar", transform=None,
                 test_mode=False, test_cfg=None, loop=1, ignore_index=255):
        super().__init__()
        self.data_root    = data_root
        self.split        = split
        self.transform    = Compose(transform)
        self.test_mode    = test_mode
        self.test_cfg     = test_cfg if test_mode else None
        self.loop         = 1 if test_mode else loop
        self.ignore_index = ignore_index

        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")

        self.data_list = sorted([
            os.path.join(split_dir, d)
            for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])

        if len(self.data_list) == 0:
            raise ValueError(f"No scene folders found in {split_dir}")

        print(f"[LidarDatasetV2][{split.upper()}] {len(self.data_list)} scenes in {split_dir}")

    def __len__(self):
        return len(self.data_list) * self.loop

    def get_data(self, idx):
        scene_dir = self.data_list[idx % len(self.data_list)]

        coord   = np.load(os.path.join(scene_dir, "coord.npy")).astype(np.float32)
        color   = np.load(os.path.join(scene_dir, "color.npy")).astype(np.float32)
        segment = np.load(os.path.join(scene_dir, "segment.npy")).astype(np.int64)
        v1_pred = np.load(os.path.join(scene_dir, "v1_pred.npy")).astype(np.float32)

        if color.ndim == 1:
            color = color.reshape(-1, 1)

        # Normalize v1_pred to [0, 1] so it has same scale as intensity
        # Max class ID is 9, so divide by 9
        v1_pred_norm = (v1_pred / 9.0).reshape(-1, 1)

        # Concatenate intensity + v1_pred -> (N, 2)
        color = np.concatenate([color, v1_pred_norm], axis=1)

        valid_max = len(self.class_names) - 1
        segment[(segment < 0) | (segment > valid_max)] = self.ignore_index

        return dict(
            coord   = coord,
            color   = color,      # (N, 2): intensity + v1_pred_normalized
            segment = segment,
            name    = os.path.basename(scene_dir),
        )

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        return self.prepare_train_data(idx)

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        assert idx < len(self.data_list)
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict
