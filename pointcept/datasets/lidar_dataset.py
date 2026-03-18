"""
pointcept/datasets/lidar_dataset.py

Custom dataset for static LiDAR tiles preprocessed from LAZ files.
Place this file in: pointcept/datasets/lidar_dataset.py
Then add the following line to pointcept/datasets/__init__.py:
    from .lidar_dataset import *
"""

import os
import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class LidarDataset(Dataset):
    """
    Dataset for static LiDAR tiles in Pointcept numpy format.

    Expected folder structure:
        data_root/
            train/
                tile_001/
                    coord.npy       (N, 3) float32
                    color.npy       (N, 1) float32  ← intensity
                    segment.npy     (N,)   int16
                tile_002/
                    ...
            val/
                ...

    Class IDs after remapping (class 8 RIEN → 255 ignore):
        0  Unclassified (ignore)
        1  Sol
        2  Poteau
        3  Panneau
        4  Potelet
        5  Tronc
        6  Vegetation
        7  Batiment
        8  Mur
        9  Barriere
        255 Ignore (RIEN + anything unmapped)

    num_classes = 10 (IDs 0-9, with 0 treated as ignore in loss)
    ignore_index = 255
    """

    class_names = [
        'Unclassified',  # 0  — typically ignored in loss
        'Sol',           # 1
        'Poteau',        # 2
        'Panneau',       # 3
        'Potelet',       # 4
        'Tronc',         # 5
        'Vegetation',    # 6
        'Batiment',      # 7
        'Mur',           # 8
        'Barriere',      # 9
    ]

    def __init__(
        self,
        split='train',
        data_root='data/lidar',
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=10,
        ignore_index=255,
    ):
        super().__init__()
        self.data_root   = data_root
        self.split       = split
        self.transform   = Compose(transform)
        self.test_mode   = test_mode
        self.test_cfg    = test_cfg if test_mode else None
        self.loop        = 1 if test_mode else loop
        self.ignore_index = ignore_index

        # Build list of scene folders
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")

        self.data_list = sorted([
            os.path.join(split_dir, d)
            for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])

        if len(self.data_list) == 0:
            raise ValueError(
                f"No scene folders found in {split_dir}. "
                f"Run data_preprocessing.py first."
            )

        print(f"[LidarDataset][{split.upper()}] "
              f"Found {len(self.data_list)} scenes in {split_dir}")

    def __len__(self):
        return len(self.data_list) * self.loop

    def get_data(self, idx):
        """Load and return raw data dict for scene at index idx."""
        scene_dir = self.data_list[idx % len(self.data_list)]

        coord   = np.load(os.path.join(scene_dir, 'coord.npy')).astype(np.float32)
        color   = np.load(os.path.join(scene_dir, 'color.npy')).astype(np.float32)
        segment = np.load(os.path.join(scene_dir, 'segment.npy')).astype(np.int64)

        # Ensure color is (N, C)
        if color.ndim == 1:
            color = color.reshape(-1, 1)

        # Clamp any out-of-range labels to ignore_index
        valid_max = len(self.class_names) - 1  # 9
        segment[(segment < 0) | (segment > valid_max)] = self.ignore_index

        return dict(
            coord   = coord,
            color    = color,          # Pointcept standard key for features
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
        """
        Test mode: return all points with voxel sliding augmentation
        so that every point gets predicted at least once.
        Mirrors the pattern used by ScanNet/S3DIS test datasets.
        """
        assert idx < len(self.data_list), \
            f"Test index {idx} out of range ({len(self.data_list)} scenes)"

        data_dict = self.get_data(idx)

        # In test mode, Pointcept's GridSampling transform handles
        # the sliding window voxelisation internally via test_cfg.
        # We just need to pass the raw data through the transform.
        data_dict = self.transform(data_dict)
        return data_dict