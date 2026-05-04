# ScanNet-1500 dataset loader modified from original to match MegaDepth1500 dataset loader

import os

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


def get_relative_transform(pose0, pose1):
    R0 = pose0[:3, :3]  # 3x3
    t0 = pose0[:3, [3]]  # 3x1

    R1 = pose1[:3, :3]  # 3x3
    t1 = pose1[:3, [3]]  # 3x1
    
    R_0to1 = R1.T @ R0  # 3x3
    t_0to1 = R1.T @ (t0 - t1)  # 3x1
    T_0to1 = np.concatenate([R_0to1, t_0to1], axis=-1)  # 3x4
    
    return T_0to1


class ScanNet1500(Dataset):
    def __init__(self, npz_file, root_dir="/shared/workspace/vicos/accelerated_features/datasets/ScanNet1500", cache_images=True):
        self.root_dir = root_dir
        self.npz_file = npz_file
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None
        self.intrinsics_cache = {}  # Cache intrinsics per scene

        if not os.path.exists(self.root_dir):
            raise RuntimeError(
                f"Dataset {self.root_dir} does not exist!")
        
        self.pairs = self._load_pairs()

    def _load_pairs(self):
        pairs = []
        gt_poses = np.load(self.npz_file)
        names = gt_poses['name']
        
        for i in range(len(names)):
            scene_id = names[i, 0]
            scene_idx = names[i, 1]
            scene = f'scene{scene_id:04d}_{scene_idx:02d}'
        
            image0 = str(int(names[i, 2]))
            image1 = str(int(names[i, 3]))
            
            if scene not in self.intrinsics_cache:
                self.intrinsics_cache[scene] = np.loadtxt(
                    os.path.join(self.root_dir, scene, 'intrinsic/intrinsic_color.txt')
                )
            K0 = self.intrinsics_cache[scene]
            K1 = K0

            pose_0 = np.loadtxt(
                os.path.join(self.root_dir, scene, 'pose', image0 + '.txt')
            )
            pose_1 = np.loadtxt(
                os.path.join(self.root_dir, scene, 'pose', image1 + '.txt')
            )
            T_0to1 = get_relative_transform(pose_0, pose_1)
            
            pairs.append({
                'image0_path': os.path.join(self.root_dir, scene, 'color', image0 + '.jpg'),
                'image1_path': os.path.join(self.root_dir, scene, 'color', image1 + '.jpg'),
                'K0': K0,
                'K1': K1,
                'T_0to1': torch.from_numpy(T_0to1).float(),  # Convert once during init
                'scene_id': scene,
                'pair_id': i,
            })

        return pairs

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        if self.image_cache is not None:
            if path not in self.image_cache:
                self.image_cache[path] = cv2.imread(path)
            return self.image_cache[path].copy()
        else:
            return cv2.imread(path)

    def __getitem__(self, idx):
        data = self.pairs[idx].copy()
        
        image0 = self._load_image(data['image0_path'])
        image1 = self._load_image(data['image1_path'])
        
        data['image0'] = torch.tensor(image0.astype(np.float32) / 255).permute(2, 0, 1)
        data['image1'] = torch.tensor(image1.astype(np.float32) / 255).permute(2, 0, 1)
        
        data['K0'] = torch.tensor(data['K0']).float()
        data['K1'] = torch.tensor(data['K1']).float()
        
        return data