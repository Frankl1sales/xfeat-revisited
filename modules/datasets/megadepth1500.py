# Megadepth-1500 dataset loader modified from original

import os
import json

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class MegaDepth1500(Dataset):
    def __init__(self, json_file, root_dir="/shared/workspace/vicos/accelerated_features/datasets/Mega1500", cache_images=True):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.root_dir = root_dir
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None

        if not os.path.exists(self.root_dir):
            raise RuntimeError(
            f"Dataset {self.root_dir} does not exist!")

    def __len__(self):
        return len(self.data)

    def _load_and_resize_image(self, path, target_size):
        if self.image_cache is not None:
            cache_key = (path, target_size)
            if cache_key not in self.image_cache:
                img = cv2.imread(f"{self.root_dir}/{path}")
                img = cv2.resize(img, target_size)
                self.image_cache[cache_key] = img
            return self.image_cache[cache_key].copy()
        else:
            img = cv2.imread(f"{self.root_dir}/{path}")
            return cv2.resize(img, target_size)

    def __getitem__(self, idx):
        data = self.data[idx].copy()

        h1, w1 = data['size0_hw']
        h2, w2 = data['size1_hw']

        image0 = self._load_and_resize_image(data['pair_names'][0], (w1, h1))
        image1 = self._load_and_resize_image(data['pair_names'][1], (w2, h2))

        data['image0'] = torch.tensor(image0.astype(np.float32) / 255).permute(2, 0, 1)
        data['image1'] = torch.tensor(image1.astype(np.float32) / 255).permute(2, 0, 1)

        for k, v in data.items():
            if k not in ('dataset_name', 'scene_id', 'pair_id', 'pair_names', 'size0_hw', 'size1_hw', 'image0', 'image1'):
                data[k] = torch.tensor(np.array(v, dtype=np.float32))

        return data