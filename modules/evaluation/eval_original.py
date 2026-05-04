# Evaluation script for original XFeat pretrained model

import argparse
import os
import sys
import json
import csv
import random
import importlib.util

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import poselib
from tqdm import tqdm

# Import datasets from reproduction (compatible)
reproduction_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, reproduction_path)
from modules.datasets.megadepth1500 import MegaDepth1500
from modules.datasets.scannet1500 import ScanNet1500
sys.path.remove(reproduction_path)

# Load original modules using importlib to avoid namespace conflicts
original_modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../original/modules'))

# Load original model.py
model_spec = importlib.util.spec_from_file_location("original_model", os.path.join(original_modules_path, "model.py"))
original_model = importlib.util.module_from_spec(model_spec)
sys.modules['original_model'] = original_model
model_spec.loader.exec_module(original_model)

# Load original interpolator.py
interpolator_spec = importlib.util.spec_from_file_location("original_interpolator", os.path.join(original_modules_path, "interpolator.py"))
original_interpolator = importlib.util.module_from_spec(interpolator_spec)
sys.modules['original_interpolator'] = original_interpolator
interpolator_spec.loader.exec_module(original_interpolator)

# Temporarily inject original modules into the modules namespace
original_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../original'))
sys.path.insert(0, original_path)

# Temporarily replace modules.model and modules.interpolator
sys.modules['modules.model'] = original_model
sys.modules['modules.interpolator'] = original_interpolator

# Now import XFeat which will use our injected modules
from modules.xfeat import XFeat

# Clean up path
sys.path.remove(original_path)

np.set_printoptions(suppress=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate original XFeat pretrained model on pose estimation benchmarks"
    )
    parser.add_argument('--dataset', type=str, choices=['megadepth1500', 'scannet1500'], required=True)
    parser.add_argument('--data-root', type=str, required=False)
    parser.add_argument('--matcher', type=str, choices=['xfeat', 'xfeat-star'], default='xfeat')
    parser.add_argument('--ransac-thr', type=float, default=2.5)
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cache-images', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.data_root is None:
        if args.dataset == 'megadepth1500':
            args.data_root = '/shared/workspace/vicos/accelerated_features/datasets/Mega1500'
        elif args.dataset == 'scannet1500':
            args.data_root = '/shared/workspace/vicos/accelerated_features/datasets/ScanNet1500'
    return args

def intrinsics_to_camera(K):
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }

def estimate_pose_poselib(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    M, info = poselib.estimate_relative_pose(
        kpts0.tolist(), 
        kpts1.tolist(),
        intrinsics_to_camera(K0),
        intrinsics_to_camera(K1),
        {
            "max_epipolar_error": thresh,
            "success_prob": conf,
            "min_iterations": 20,
            "max_iterations": 10_000,
        },
    )
    return (M.R, M.t, np.array(info["inliers"]))

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    
    # Handle zero norm case
    if n < 1e-10 or np.linalg.norm(t_gt) < ignore_gt_t_thr:
        t_err = 0.0
    else:
        t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
        t_err = np.minimum(t_err, 180 - t_err)

    R_gt = T_0to1[:3, :3]
    cos = np.clip((np.trace(np.dot(R.T, R_gt)) - 1) / 2, -1., 1.)
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err

def compute_pose_error(pair, ransac_thr=2.5):
    def to_numpy(x):
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        return x.squeeze() if x.ndim == 3 else x
    pts0 = to_numpy(pair['pts0'])
    pts1 = to_numpy(pair['pts1'])
    K0 = to_numpy(pair['K0'])
    K1 = to_numpy(pair['K1'])
    T_0to1 = to_numpy(pair['T_0to1'])
    if len(pts0) < 5:
        return {'R_err': np.inf, 't_err': np.inf, 'inliers': []}
    ret = estimate_pose_poselib(pts0, pts1, K0, K1, ransac_thr)
    if ret is not None:
        R, t, inliers = ret
        t_err, R_err = relative_pose_error(T_0to1, R, t)
        return {'R_err': R_err, 't_err': t_err, 'inliers': inliers}
    else:
        return {'R_err': np.inf, 't_err': np.inf, 'inliers': []}

def error_auc(errors, thresholds=[5, 10, 20]):
    """Compute AUC with standard error estimation.
    
    SE is computed using the variance of binary indicators (error <= threshold)
    for each threshold, then propagated through the integration.
    """
    n = len(errors)
    errors_sorted = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors_sorted)))
    aucs = []
    ses = []
    
    for thr in thresholds:
        last_index = np.searchsorted(errors_sorted, thr)
        y = recall[:last_index] + [recall[last_index-1]] if last_index > 0 else [0, 0]
        x = errors_sorted[:last_index] + [thr] if last_index > 0 else [0, thr]
        auc = np.trapezoid(y, x) / thr
        aucs.append(auc)
        
        # Compute SE: treat AUC as proportion of samples below threshold, weighted by error
        # Simplified approach: SE based on success rate variance
        success_rate = np.mean(errors <= thr)
        se = np.sqrt(success_rate * (1 - success_rate) / n) if n > 0 else 0.0
        ses.append(se)
    
    results = {}
    for t, auc, se in zip(thresholds, aucs, ses):
        results[f'auc@{t}'] = auc
        results[f'auc@{t}_se'] = se
    return results

def compute_metrics(pairs, dataset):
    print(f"\nEvaluating on {len(pairs)} pairs")
    print("-" * 50)
    n = len(pairs)
    errors = [max(p['t_err'], p['R_err']) for p in pairs]
    errors = np.array(errors)
    
    if dataset == 'megadepth1500':
        auc_results = error_auc(errors, [5, 10, 20])
        
        acc10 = float((errors <= 10).mean())
        acc10_se = float(np.sqrt(acc10 * (1 - acc10) / n))
        
        inlier_ratios = []
        inlier_counts = []
        match_counts = []
        for p in pairs:
            inlier_mask = p.get('inliers', [])
            n_inliers = np.count_nonzero(inlier_mask)
            n_matches = len(p.get('pts0', []))
            inlier_counts.append(n_inliers)
            inlier_ratios.append(n_inliers / n_matches if n_matches > 0 else 0)
            match_counts.append(n_matches)
        
        mir = float(np.mean(inlier_ratios))
        mir_se = float(np.std(inlier_ratios, ddof=1) / np.sqrt(n))
        
        mean_inliers = float(np.mean(inlier_counts))
        mean_inliers_se = float(np.std(inlier_counts, ddof=1) / np.sqrt(n))
        
        mean_matches = float(np.mean(match_counts))
        
        metrics = dict(auc_results)
        metrics['Acc@10'] = acc10
        metrics['Acc@10_se'] = acc10_se
        metrics['MIR'] = mir
        metrics['MIR_se'] = mir_se
        metrics['mean_inliers'] = mean_inliers
        metrics['mean_inliers_se'] = mean_inliers_se
        
        print("\nAUC Metrics (with SE):")
        for thr in [5, 10, 20]:
            auc_key = f'auc@{thr}'
            se_key = f'auc@{thr}_se'
            if auc_key in auc_results:
                print(f"  {auc_key}: {auc_results[auc_key]*100:.2f}% ± {auc_results[se_key]*100:.2f}%")
        print(f"\nAcc@10°: {acc10*100:.2f}% ± {acc10_se*100:.2f}%")
        print(f"Mean Inlier Ratio (MIR): {mir*100:.2f}% ± {mir_se*100:.2f}%")
        print(f"Mean #Inliers: {mean_inliers:.2f} ± {mean_inliers_se:.2f}")
        
    elif dataset == 'scannet1500':
        auc_results = error_auc(errors, [5, 10, 20])
        metrics = dict(auc_results)
        
        print("\nAUC Metrics (with SE):")
        for thr in [5, 10, 20]:
            auc_key = f'auc@{thr}'
            se_key = f'auc@{thr}_se'
            if auc_key in auc_results:
                print(f"  {auc_key}: {auc_results[auc_key]*100:.2f}% ± {auc_results[se_key]*100:.2f}%")

def save_predictions_csv(pairs, output_file):
    """Save detailed predictions to CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_idx', 'num_matches', 't_err', 'R_err', 'pose_err', 'passed'])
        
        for idx, p in enumerate(pairs):
            num_matches = len(p.get('pts0', []))
            t_err = p.get('t_err', np.inf)
            R_err = p.get('R_err', np.inf)
            pose_err = max(t_err, R_err)
            passed = 1 if pose_err <= 10 else 0
            
            writer.writerow([idx, num_matches, f"{t_err:.4f}", f"{R_err:.4f}", f"{pose_err:.4f}", passed])
    print(f"Predictions saved to {output_file}")

def tensor2bgr(t):
    if torch.is_tensor(t):
        arr = t.numpy()
    else:
        arr = t
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def evaluate_dataset(matcher_fn, dataset, ransac_thr=0.5, batch_size=1, num_workers=0, seed=42):
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    pairs = []
    n_no_matches = 0
    for d in tqdm(loader, desc="Evaluating"):
        img0 = tensor2bgr(d['image0'])
        img1 = tensor2bgr(d['image1'])
        mkpts0, mkpts1 = matcher_fn(img0, img1)
        if len(mkpts0) == 0:
            n_no_matches += 1
        if 'scale0' in d:
            mkpts0 *= d['scale0'].cpu().numpy()
        if 'scale1' in d:
            mkpts1 *= d['scale1'].cpu().numpy()
        d['pts0'], d['pts1'] = mkpts0, mkpts1
        d.update(compute_pose_error(d, ransac_thr))
        pairs.append(d)
    if n_no_matches > 0:
        print(f"\nWarning: {n_no_matches} pairs had no matches!")
    return pairs

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed set to: {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'megadepth1500':
        json_file = 'assets/megadepth_1500.json'
        dataset = MegaDepth1500(
            json_file=json_file,
            root_dir=os.path.join(args.data_root, 'megadepth_test_1500'),
            cache_images=args.cache_images
        )
    elif args.dataset == 'scannet1500':
        dataset = ScanNet1500(
            npz_file=os.path.join(args.data_root, 'test.npz'),
            root_dir=os.path.join(args.data_root, 'scannet_test_1500'),
            cache_images=args.cache_images
        )
    print(f"Dataset: {len(dataset)} pairs" + (" (cached)" if args.cache_images else ""))
    
    weights_path = 'weights/xfeat.pt'
    top_k = args.top_k if args.top_k is not None else (10000 if args.matcher == 'xfeat-star' else 4096)
    xfeat = XFeat(weights=weights_path, top_k=top_k)
    matcher_fn = xfeat.match_xfeat_star if args.matcher == 'xfeat-star' else xfeat.match_xfeat
    print(f"XFeat top_k: {xfeat.top_k}")
    
    print(f"\nEvaluating (RANSAC threshold: {args.ransac_thr}px, workers: {args.num_workers})")
    with torch.no_grad():
        pairs = evaluate_dataset(matcher_fn, dataset, args.ransac_thr, num_workers=args.num_workers, seed=args.seed)
    compute_metrics(pairs, args.dataset)
    
    # Save predictions to CSV
    csv_file = os.path.join(args.output_dir, f"{args.dataset}_{args.matcher}_original_predictions.csv")
    save_predictions_csv(pairs, csv_file)
    
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.matcher}_original_ransac{args.ransac_thr}.json")
    errors = np.array([max(p['t_err'], p['R_err']) for p in pairs])
    n = len(pairs)
    
    if args.dataset == 'megadepth1500':
        auc_results = error_auc(errors, [5, 10, 20])
        
        acc10 = float((errors <= 10).mean())
        acc10_se = float(np.sqrt(acc10 * (1 - acc10) / n))
        
        inlier_ratios = []
        inlier_counts = []
        for p in pairs:
            inlier_mask = p.get('inliers', [])
            n_inliers = np.count_nonzero(inlier_mask)
            n_matches = len(p.get('pts0', []))
            inlier_counts.append(n_inliers)
            inlier_ratios.append(n_inliers / n_matches if n_matches > 0 else 0)
        
        mir = float(np.mean(inlier_ratios))
        mir_se = float(np.std(inlier_ratios, ddof=1) / np.sqrt(n))
        
        mean_inliers = float(np.mean(inlier_counts))
        mean_inliers_se = float(np.std(inlier_counts, ddof=1) / np.sqrt(n))
        
        metrics = dict(auc_results)
        metrics['Acc@10'] = acc10
        metrics['Acc@10_se'] = acc10_se
        metrics['MIR'] = mir
        metrics['MIR_se'] = mir_se
        metrics['mean_inliers'] = mean_inliers
        metrics['mean_inliers_se'] = mean_inliers_se
        
    elif args.dataset == 'scannet1500':
        metrics = error_auc(errors, [5, 10, 20])
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'matcher': args.matcher,
            'model': 'original_pretrained',
            'seed': args.seed,
            'ransac_thr': args.ransac_thr,
            'num_pairs': len(pairs),
            'metrics': metrics
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main()
