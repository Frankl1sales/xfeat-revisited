import argparse
import os
import sys
import time
import json
import csv
import subprocess
import numpy as np
import torch
import cv2
import poselib
from tqdm import tqdm

from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not os.path.exists(os.path.join(REPO_ROOT, 'modules')):
    raise RuntimeError(f"Could not find 'modules/' under repository root: {REPO_ROOT}")

if os.path.abspath(os.getcwd()) != REPO_ROOT:
    raise RuntimeError(f"Run this module from the repository root: {REPO_ROOT}")

from modules.xfeat import XFeat
from modules.xfeat_original import XFeat as XFeatOriginal


class MegaDepth1500(torch.utils.data.Dataset):
    def __init__(self, json_file, root_dir="datasets/Mega1500", cache_images=True):
        with open(json_file, 'r') as handle:
            self.data = json.load(handle)

        self.root_dir = root_dir
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None

        if not os.path.exists(self.root_dir):
            raise RuntimeError(f"Dataset {self.root_dir} does not exist!")

    def __len__(self):
        return len(self.data)

    def _load_and_resize_image(self, path, target_size):
        full_path = os.path.join(self.root_dir, path)
        if self.image_cache is not None:
            cache_key = (full_path, target_size)
            if cache_key not in self.image_cache:
                img = cv2.imread(full_path)
                if img is None:
                    raise RuntimeError(f"Could not read image: {full_path}")
                img = cv2.resize(img, target_size)
                self.image_cache[cache_key] = img
            return self.image_cache[cache_key].copy()

        img = cv2.imread(full_path)
        if img is None:
            raise RuntimeError(f"Could not read image: {full_path}")
        return cv2.resize(img, target_size)

    def __getitem__(self, idx):
        sample = self.data[idx].copy()

        h0, w0 = sample['size0_hw']
        h1, w1 = sample['size1_hw']

        image0 = self._load_and_resize_image(sample['pair_names'][0], (w0, h0))
        image1 = self._load_and_resize_image(sample['pair_names'][1], (w1, h1))

        sample['image0'] = torch.tensor(image0.astype(np.float32) / 255.0).permute(2, 0, 1)
        sample['image1'] = torch.tensor(image1.astype(np.float32) / 255.0).permute(2, 0, 1)

        for key, value in list(sample.items()):
            if key not in ('dataset_name', 'scene_id', 'pair_id', 'pair_names', 'size0_hw', 'size1_hw', 'image0', 'image1'):
                sample[key] = torch.tensor(np.array(value, dtype=np.float32))

        return sample

def load_xfeat():
    # Try to find weights in common locations
    weights_candidates = [
        os.path.join(REPO_ROOT, 'checkpoints', 'xfeat_default_skip_default.pth'),
        os.path.join(REPO_ROOT, 'checkpoints', 'xfeat.pth'),
        os.path.join(REPO_ROOT, 'xfeat.pth'),
        'xfeat.pth',
    ]
    weights_path = None
    for candidate in weights_candidates:
        if os.path.exists(candidate):
            weights_path = candidate
            break
    return XFeat(weights=weights_path, top_k=4096)

def load_xfeat_original():
    # Try to find weights in common locations
    weights_candidates = [
        os.path.join(REPO_ROOT, 'checkpoints', 'xfeat.pt'),
        os.path.join(REPO_ROOT, 'xfeat.pt'),
        'xfeat.pt',
    ]
    weights_path = None
    for candidate in weights_candidates:
        if os.path.exists(candidate):
            weights_path = candidate
            break
    return XFeatOriginal(weights=weights_path, top_k=4096)

def load_xfeat_star():
    # Try to find weights in common locations
    weights_candidates = [
        os.path.join(REPO_ROOT, 'checkpoints', 'xfeat_default_skip_default.pth'),
        os.path.join(REPO_ROOT, 'checkpoints', 'xfeat.pth'),
        os.path.join(REPO_ROOT, 'xfeat.pth'),
        'xfeat.pth',
    ]
    weights_path = None
    for candidate in weights_candidates:
        if os.path.exists(candidate):
            weights_path = candidate
            break
    return XFeat(weights=weights_path, top_k=10000)

def load_xfeat_star_original():
    # Try to find weights in common locations
    weights_candidates = [
        os.path.join(REPO_ROOT, 'checkpoints', 'xfeat.pt'),
        os.path.join(REPO_ROOT, 'xfeat.pt'),
        os.path.join(REPO_ROOT, 'weights', 'xfeat.pt'),
        'xfeat.pt',
    ]
    weights_path = None
    for candidate in weights_candidates:
        if os.path.exists(candidate):
            weights_path = candidate
            break
    return XFeatOriginal(weights=weights_path, top_k=10000)

def clone_submodule_if_missing(repo_url, target_dir, branch=None):
    if not os.path.exists(target_dir):
        print(f"Cloning {repo_url} into {target_dir}...")
        clone_cmd = ['git', 'clone', repo_url, target_dir]
        if branch:
            clone_cmd.extend(['-b', branch])
        subprocess.run(clone_cmd, check=True)

def load_disk():
    disk_dir = os.path.join(REPO_ROOT, 'timing', 'disk')
    if not os.path.exists(disk_dir):
        clone_submodule_if_missing("https://github.com/cvlab-epfl/disk.git", disk_dir)
        subprocess.run(['git', '-C', disk_dir, 'submodule', 'update', '--init', '--recursive'], check=True)
    sys.path.insert(0, disk_dir)
    # Only add submodules to sys.path if they exist
    submodules = [
        os.path.join(disk_dir, "submodules/torch-dimcheck"),
        os.path.join(disk_dir, "submodules/torch-localize"),
        os.path.join(disk_dir, "submodules/unets"),
    ]
    for path in submodules:
        if os.path.exists(path):
            sys.path.insert(0, path)
    from disk.common.structs import Features, NpArray
    from disk.model.disk import DISK
    weights_path = os.path.join(disk_dir, "depth-save.pth")
    model = DISK()
    state_dict = torch.load(weights_path, map_location="cpu")
    if 'extractor' in state_dict:
        state_dict = state_dict['extractor']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def load_alike():
    alike_dir = os.path.join(REPO_ROOT, 'third_party', 'ALIKE')
    if not os.path.exists(alike_dir):
        clone_submodule_if_missing("https://github.com/Shiaoming/ALIKE.git", alike_dir)
    sys.path.insert(0, alike_dir)
    from alike import ALike
    weights_path = os.path.join(alike_dir, "models/alike-t.pth")
    model = ALike(
        c1=8, c2=16, c3=32, c4=64, dim=64, single_head=True, radius=2,
        model_path=weights_path,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        top_k=4096, scores_th=0.1, n_limit=4096
    )
    return model

def load_superpoint():
    superpoint_dir = os.path.join(REPO_ROOT, 'timing', 'SuperPoint')
    if not os.path.exists(superpoint_dir):
        clone_submodule_if_missing("https://github.com/rpautrat/SuperPoint.git", superpoint_dir)
    sys.path.insert(0, superpoint_dir)
    from superpoint_pytorch import SuperPoint
    weights_path = os.path.join(superpoint_dir, "weights/superpoint_v6_from_tf.pth")
    model = SuperPoint(max_num_keypoints=4096)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_orb():
    return cv2.ORB_create(nfeatures=4096)

METHODS = [
    {"name": "XFeat", "loader": load_xfeat, "method_type": "xfeat"},
    {"name": "XFeat (Original)", "loader": load_xfeat_original, "method_type": "xfeat_original"},
    {"name": "XFeat*", "loader": load_xfeat_star, "method_type": "xfeat_star"},
    {"name": "XFeat* (Original)", "loader": load_xfeat_star_original, "method_type": "xfeat_star_original"},
    {"name": "ALIKE", "loader": load_alike, "method_type": "alike"},
    {"name": "DISK", "loader": load_disk, "method_type": "disk"},
    {"name": "ORB", "loader": load_orb, "method_type": "orb"},
    {"name": "SuperPoint", "loader": load_superpoint, "method_type": "superpoint"},
]

DATASET_JSON = os.path.join(REPO_ROOT, 'assets', 'megadepth_1500.json')
DATASET_ROOT = os.path.join(REPO_ROOT, 'datasets', 'Mega1500', 'megadepth_test_1500')
DATASET_BATCH_SIZE = 1
RESULTS_CSV = os.path.join(os.path.dirname(__file__), 'benchmark_results.csv')
PREDICTIONS_CSV = os.path.join(os.path.dirname(__file__), 'predictions.csv')

def load_megadepth_dataloader(json_path=DATASET_JSON, root_dir=DATASET_ROOT, batch_size=DATASET_BATCH_SIZE, cache_images=True):
    dataset = MegaDepth1500(json_file=json_path, root_dir=root_dir, cache_images=cache_images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def run_experiment(method, max_native_samples=None, max_vga_samples=None):
    model = method["loader"]()
    method_type = method.get("method_type", "")
    dataloader = load_megadepth_dataloader()
    acc10_list = []
    predictions_list = []
    
    print(f"  Stage 1: Computing accuracy at native resolution...")
    for sample_idx, data in enumerate(tqdm(dataloader, desc=f"  Accuracy", leave=False)):
        if max_native_samples is not None and sample_idx >= max_native_samples:
            break

        img0 = data['image0'][0].numpy().transpose(1, 2, 0)
        img1 = data['image1'][0].numpy().transpose(1, 2, 0)
        img0 = (img0 * 255).clip(0, 255).astype(np.uint8)
        img1 = (img1 * 255).clip(0, 255).astype(np.uint8)

        if method_type == 'xfeat_star':
            # XFeat*
            mkpts0, mkpts1 = model.match_xfeat_star(img0, img1)
        elif hasattr(model, 'match_xfeat'):
            # XFeat
            mkpts0, mkpts1 = model.match_xfeat(img0, img1)
        elif hasattr(model, 'detectAndCompute'):
            # ORB
            kp1, desc1 = model.detectAndCompute(cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY), None)
            kp2, desc2 = model.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), None)
            if desc1 is None or desc2 is None or len(kp1) < 5 or len(kp2) < 5:
                acc10_list.append(0)
                predictions_list.append({"sample_idx": sample_idx, "method": method["name"], "num_matches": 0, "t_err": None, "R_err": None, "pose_err": None, "passed": 0})
                continue
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc1, desc2)
            if len(matches) < 5:
                acc10_list.append(0)
                predictions_list.append({"sample_idx": sample_idx, "method": method["name"], "num_matches": len(matches), "t_err": None, "R_err": None, "pose_err": None, "passed": 0})
                continue
            mkpts0 = np.array([kp1[m.queryIdx].pt for m in matches])
            mkpts1 = np.array([kp2[m.trainIdx].pt for m in matches])
        elif hasattr(model, 'features'):
            # DISK
            img0_t = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            feats0 = model.features(img0_t, kind='nms', window_size=5, cutoff=0.0, n=4096)[0]
            feats1 = model.features(img1_t, kind='nms', window_size=5, cutoff=0.0, n=4096)[0]
            if len(feats0.kp) < 5 or len(feats1.kp) < 5:
                acc10_list.append(0)
                predictions_list.append({"sample_idx": sample_idx, "method": method["name"], "num_matches": 0, "t_err": None, "R_err": None, "pose_err": None, "passed": 0})
                continue
            desc0 = feats0.desc if torch.is_tensor(feats0.desc) else torch.from_numpy(feats0.desc).float()
            desc1 = feats1.desc if torch.is_tensor(feats1.desc) else torch.from_numpy(feats1.desc).float()
            dmat = torch.cdist(desc0, desc1)
            idx0 = torch.arange(len(desc0))
            idx1 = torch.argmin(dmat, dim=1)
            idx1_reverse = torch.argmin(dmat, dim=0)
            mutual = idx1_reverse[idx1] == idx0
            mkpts0 = feats0.kp.cpu().numpy() if torch.is_tensor(feats0.kp) else feats0.kp
            mkpts1 = feats1.kp.cpu().numpy() if torch.is_tensor(feats1.kp) else feats1.kp
            mkpts0 = mkpts0[mutual.numpy()]
            mkpts1 = mkpts1[idx1[mutual].numpy()]
            if len(mkpts0) < 5:
                acc10_list.append(0)
                predictions_list.append({"sample_idx": sample_idx, "method": method["name"], "num_matches": len(mkpts0), "t_err": None, "R_err": None, "pose_err": None, "passed": 0})
                continue
        elif hasattr(model, '__class__') and 'ALike' in model.__class__.__name__:
            # ALIKE
            pred0 = model(img0, sub_pixel=True)
            pred1 = model(img1, sub_pixel=True)
            kpts0 = pred0['keypoints']
            kpts1 = pred1['keypoints']
            desc0 = pred0['descriptors']
            desc1 = pred1['descriptors']
            if len(kpts0) < 5 or len(kpts1) < 5:
                acc10_list.append(0)
                predictions_list.append({"sample_idx": sample_idx, "method": method["name"], "num_matches": 0, "t_err": None, "R_err": None, "pose_err": None, "passed": 0})
                continue
            desc0_norm = desc0 / (np.linalg.norm(desc0, axis=1, keepdims=True) + 1e-8)
            desc1_norm = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)
            dmat = desc0_norm @ desc1_norm.T
            idx0 = np.arange(len(desc0))
            idx1 = np.argmax(dmat, axis=1)
            idx0_reverse = np.argmax(dmat, axis=0)
            mutual = idx0_reverse[idx1] == idx0
            mkpts0 = kpts0[mutual]
            mkpts1 = kpts1[idx1[mutual]]
            if len(mkpts0) < 5:
                acc10_list.append(0)
                predictions_list.append({"sample_idx": sample_idx, "method": method["name"], "num_matches": len(mkpts0), "t_err": None, "R_err": None, "pose_err": None, "passed": 0})
                continue
        elif isinstance(model, torch.nn.Module):
            # SuperPoint
            img0_gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img0_t = torch.from_numpy(img0_gray).float()[None, None] / 255.0
            img1_t = torch.from_numpy(img1_gray).float()[None, None] / 255.0
            
            pred0 = model({"image": img0_t})
            pred1 = model({"image": img1_t})
            
            kpts0 = pred0['keypoints'][0].detach().cpu().numpy()
            kpts1 = pred1['keypoints'][0].detach().cpu().numpy()
            desc0 = pred0['descriptors'][0].detach().cpu().numpy()
            desc1 = pred1['descriptors'][0].detach().cpu().numpy()
            
            if len(kpts0) < 5 or len(kpts1) < 5:
                acc10_list.append(0)
                predictions_list.append({"sample_idx": sample_idx, "method": method["name"], "num_matches": 0, "t_err": None, "R_err": None, "pose_err": None, "passed": 0})
                continue
            
            desc0_norm = desc0 / (np.linalg.norm(desc0, axis=1, keepdims=True) + 1e-8)
            desc1_norm = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)
            dmat = desc0_norm @ desc1_norm.T
            idx0 = np.arange(len(desc0))
            idx1 = np.argmax(dmat, axis=1)
            idx0_reverse = np.argmax(dmat, axis=0)
            mutual = idx0_reverse[idx1] == idx0
            mkpts0 = kpts0[mutual]
            mkpts1 = kpts1[idx1[mutual]]
            if len(mkpts0) < 5:
                acc10_list.append(0)
                predictions_list.append({"sample_idx": sample_idx, "method": method["name"], "num_matches": len(mkpts0), "t_err": None, "R_err": None, "pose_err": None, "passed": 0})
                continue

        if 'scale0' in data:
            scale0 = data['scale0'][0].numpy()
            mkpts0 = mkpts0 * scale0
        if 'scale1' in data:
            scale1 = data['scale1'][0].numpy()
            mkpts1 = mkpts1 * scale1

        K0 = data['K0'][0].numpy()
        K1 = data['K1'][0].numpy()
        T_0to1 = data['T_0to1'][0].numpy()

        if len(mkpts0) < 5 or len(mkpts1) < 5:
            acc10_list.append(0)
            predictions_list.append({"sample_idx": sample_idx, "method": method["name"], "num_matches": len(mkpts0), "t_err": None, "R_err": None, "pose_err": None, "passed": 0})
            continue
        try:
            M, info = poselib.estimate_relative_pose(
                mkpts0.tolist(), mkpts1.tolist(),
                {"model": "PINHOLE", "width": int(K0[0,2]*2), "height": int(K0[1,2]*2), "params": [K0[0,0], K0[1,1], K0[0,2], K0[1,2]]},
                {"model": "PINHOLE", "width": int(K1[0,2]*2), "height": int(K1[1,2]*2), "params": [K1[0,0], K1[1,1], K1[0,2], K1[1,2]]},
                {"max_epipolar_error": 2.5, "success_prob": 0.99999, "min_iterations": 20, "max_iterations": 10000}
            )
            R, t = M.R, M.t
            t_gt = T_0to1[:3, 3]
            n = np.linalg.norm(t) * np.linalg.norm(t_gt)
            t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
            t_err = np.minimum(t_err, 180 - t_err)
            R_gt = T_0to1[:3, :3]
            cos = np.clip((np.trace(np.dot(R.T, R_gt)) - 1) / 2, -1., 1.)
            R_err = np.rad2deg(np.abs(np.arccos(cos)))
            err = max(t_err, R_err)
            passed = float(err <= 10)
            acc10_list.append(passed)
            predictions_list.append({"sample_idx": sample_idx, "method": method["name"], "num_matches": len(mkpts0), "t_err": float(t_err), "R_err": float(R_err), "pose_err": float(err), "passed": int(passed)})
        except Exception:
            acc10_list.append(0)
            predictions_list.append({"sample_idx": sample_idx, "method": method["name"], "num_matches": len(mkpts0) if len(mkpts0) >= 5 else 0, "t_err": None, "R_err": None, "pose_err": None, "passed": 0})
    
    acc_10deg = np.mean(acc10_list) * 100 if acc10_list else 0
    n_samples = len(acc10_list)
    p = acc_10deg / 100.0
    acc_std_err = np.sqrt(p * (1 - p) / n_samples) * 100 if n_samples > 0 else 0
    
    print(f"  Stage 2: Computing FPS at VGA resolution...")
    vga_samples = min(30, len(dataloader.dataset)) if max_vga_samples is None else min(max_vga_samples, len(dataloader.dataset))
    vga_times = []
    
    for i, data in enumerate(dataloader):
        if i >= vga_samples:
            break
        img0 = data['image0'][0].numpy().transpose(1, 2, 0)
        img1 = data['image1'][0].numpy().transpose(1, 2, 0)
        img0 = (img0 * 255).clip(0, 255).astype(np.uint8)
        img1 = (img1 * 255).clip(0, 255).astype(np.uint8)

        img0_vga = cv2.resize(img0, (640, 480))
        img1_vga = cv2.resize(img1, (640, 480))

        start = time.time()
        if method_type == 'xfeat_star':
            mkpts0, mkpts1 = model.match_xfeat_star(img0_vga, img1_vga)
        elif hasattr(model, 'match_xfeat'):
            mkpts0, mkpts1 = model.match_xfeat(img0_vga, img1_vga)
        elif hasattr(model, 'detectAndCompute'):
            kp1, desc1 = model.detectAndCompute(cv2.cvtColor(img0_vga, cv2.COLOR_RGB2GRAY), None)
            kp2, desc2 = model.detectAndCompute(cv2.cvtColor(img1_vga, cv2.COLOR_RGB2GRAY), None)
            if desc1 is not None and desc2 is not None and len(kp1) >= 5 and len(kp2) >= 5:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(desc1, desc2)
                mkpts0 = np.array([kp1[m.queryIdx].pt for m in matches])
                mkpts1 = np.array([kp2[m.trainIdx].pt for m in matches])
            else:
                mkpts0, mkpts1 = np.array([]), np.array([])
        elif hasattr(model, 'features'):
            img0_t = torch.from_numpy(img0_vga).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img1_t = torch.from_numpy(img1_vga).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            feats0 = model.features(img0_t, kind='nms', window_size=5, cutoff=0.0, n=4096)[0]
            feats1 = model.features(img1_t, kind='nms', window_size=5, cutoff=0.0, n=4096)[0]
            if len(feats0.kp) >= 5 and len(feats1.kp) >= 5:
                desc0 = feats0.desc if torch.is_tensor(feats0.desc) else torch.from_numpy(feats0.desc).float()
                desc1 = feats1.desc if torch.is_tensor(feats1.desc) else torch.from_numpy(feats1.desc).float()
                dmat = torch.cdist(desc0, desc1)
                idx0 = torch.arange(len(desc0))
                idx1 = torch.argmin(dmat, dim=1)
                idx1_reverse = torch.argmin(dmat, dim=0)
                mutual = idx1_reverse[idx1] == idx0
                mkpts0 = feats0.kp.cpu().numpy() if torch.is_tensor(feats0.kp) else feats0.kp
                mkpts1 = feats1.kp.cpu().numpy() if torch.is_tensor(feats1.kp) else feats1.kp
                mkpts0 = mkpts0[mutual.numpy()]
                mkpts1 = mkpts1[idx1[mutual].numpy()]
            else:
                mkpts0, mkpts1 = np.array([]), np.array([])
        elif hasattr(model, '__class__') and 'ALike' in model.__class__.__name__:
            pred0 = model(img0_vga, sub_pixel=True)
            pred1 = model(img1_vga, sub_pixel=True)
            kpts0, kpts1 = pred0['keypoints'], pred1['keypoints']
            desc0, desc1 = pred0['descriptors'], pred1['descriptors']
            if len(kpts0) >= 5 and len(kpts1) >= 5:
                desc0_norm = desc0 / (np.linalg.norm(desc0, axis=1, keepdims=True) + 1e-8)
                desc1_norm = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)
                dmat = desc0_norm @ desc1_norm.T
                idx0 = np.arange(len(desc0))
                idx1 = np.argmax(dmat, axis=1)
                idx0_reverse = np.argmax(dmat, axis=0)
                mutual = idx0_reverse[idx1] == idx0
                mkpts0 = kpts0[mutual]
                mkpts1 = kpts1[idx1[mutual]]
            else:
                mkpts0, mkpts1 = np.array([]), np.array([])
        elif isinstance(model, torch.nn.Module):
            img0_gray = cv2.cvtColor(img0_vga, cv2.COLOR_RGB2GRAY)
            img1_gray = cv2.cvtColor(img1_vga, cv2.COLOR_RGB2GRAY)
            img0_t = torch.from_numpy(img0_gray).float()[None, None] / 255.0
            img1_t = torch.from_numpy(img1_gray).float()[None, None] / 255.0
            pred0 = model({"image": img0_t})
            pred1 = model({"image": img1_t})
            kpts0 = pred0['keypoints'][0].detach().cpu().numpy()
            kpts1 = pred1['keypoints'][0].detach().cpu().numpy()
            desc0 = pred0['descriptors'][0].detach().cpu().numpy()
            desc1 = pred1['descriptors'][0].detach().cpu().numpy()
            if len(kpts0) >= 5 and len(kpts1) >= 5:
                desc0_norm = desc0 / (np.linalg.norm(desc0, axis=1, keepdims=True) + 1e-8)
                desc1_norm = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)
                dmat = desc0_norm @ desc1_norm.T
                idx0 = np.arange(len(desc0))
                idx1 = np.argmax(dmat, axis=1)
                idx0_reverse = np.argmax(dmat, axis=0)
                mutual = idx0_reverse[idx1] == idx0
                mkpts0 = kpts0[mutual]
                mkpts1 = kpts1[idx1[mutual]]
            else:
                mkpts0, mkpts1 = np.array([]), np.array([])
        
        if len(mkpts0) >= 5 and len(mkpts1) >= 5:
            K0 = data['K0'][0].numpy()
            K1 = data['K1'][0].numpy()
            try:
                M, info = poselib.estimate_relative_pose(
                    mkpts0.tolist(), mkpts1.tolist(),
                    {"model": "PINHOLE", "width": 640, "height": 480, "params": [K0[0,0], K0[1,1], 320.0, 240.0]},
                    {"model": "PINHOLE", "width": 640, "height": 480, "params": [K1[0,0], K1[1,1], 320.0, 240.0]},
                    {"max_epipolar_error": 2.5, "success_prob": 0.99999, "min_iterations": 20, "max_iterations": 10000}
                )
            except Exception:
                pass 
        
        vga_times.append(time.time() - start)

    if vga_times:
        fps_samples = [2.0 / t for t in vga_times]
        fps_vga = np.mean(fps_samples)
        fps_std = np.std(fps_samples)
    else:
        fps_vga = 0
        fps_std = 0

    return acc_10deg, acc_std_err, fps_vga, fps_std, predictions_list

def main():
    parser = argparse.ArgumentParser(description="Timing benchmark runner.")
    parser.add_argument('--smoke-test', action='store_true', help='Run one sample through each method to check dependencies.')
    args = parser.parse_args()

    if args.smoke_test:
        print('Running smoke-test: loading methods and benchmarking one sample...')
        for method in METHODS:
            method_name = method['name']
            try:
                acc, acc_se, fps, fps_std, _ = run_experiment(method, max_native_samples=1, max_vga_samples=1)
                print(f'  OK: {method_name} -> Accuracy@10°={acc:.2f}%±{acc_se:.2f}%, FPS@VGA={fps:.2f}±{fps_std:.2f}')
            except Exception as exc:
                print(f'  FAIL: {method_name} -> {exc}')
                raise
        print('Smoke-test complete.')
        return

    print(f"Running benchmark...\n")
    
    for method in METHODS:
        method_name = method['name']
        print(f"Benchmarking {method_name}...")
        try:
            acc, acc_se, fps, fps_std, predictions = run_experiment(method)
            print(f"  {method_name}: Accuracy@10°={acc:.2f}%±{acc_se:.2f}%, FPS@VGA={fps:.2f}±{fps_std:.2f}\n")
            
            file_exists = os.path.exists(PREDICTIONS_CSV)
            with open(PREDICTIONS_CSV, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["sample_idx", "method", "num_matches", "t_err", "R_err", "pose_err", "passed"])
                if not file_exists:
                    writer.writeheader()
                writer.writerows(predictions)
            print(f"  Predictions for {method_name} saved to {PREDICTIONS_CSV}")
            
            file_exists = os.path.exists(RESULTS_CSV)
            with open(RESULTS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Method", "PoseAccuracy@10deg", "Accuracy_SE", "FPS_VGA", "FPS_VGA_STD"])
                writer.writerow([method_name, round(acc, 2), round(acc_se, 2), round(fps, 2), round(fps_std, 2)])
            print(f"  Results appended to {RESULTS_CSV}\n")
            
        except Exception as e:
            print(f"  ERROR: {method_name} failed with: {e}")
            print(f"  Skipping {method_name} and continuing with next method...\n")
            continue
    
    print(f"\nBenchmark complete! Final results in {RESULTS_CSV} and {PREDICTIONS_CSV}")

if __name__ == "__main__":
    main()