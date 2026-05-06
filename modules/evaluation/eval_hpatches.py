import argparse
import os
import glob
import numpy as np
import cv2
import torch
from tqdm import tqdm

from modules.xfeat import XFeat
from modules.xfeat_original import XFeat as XFeat_original

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate XFeat Homography estimation on HPatches"
    )
    parser.add_argument('--data-root', type=str, help='Path to hpatches-sequences-release', default='datasets/hpatches-sequences-release')
    parser.add_argument('--matcher', type=str, choices=['xfeat', 'xfeat-star'], default='xfeat')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint')
    parser.add_argument('--ransac-thr', type=float, default=1.5, help='MAGSAC++ threshold')
    parser.add_argument('--top-k', type=int, default=4096)
    parser.add_argument('--exp', type=str, default='skip_default')
    parser.add_argument('--use-original', action='store_true')
    args = parser.parse_args()
    return args

def compute_homography_error(H_est, H_gt, img_shape):
    h, w = img_shape
    corners = np.array([
        [0, 0, 1],[w - 1, 0, 1],
        [0, h - 1, 1],[w - 1, h - 1, 1]
    ]).T

    # Transform corners with ground truth homography
    gt_corners = H_gt @ corners
    gt_corners = gt_corners[:2, :] / gt_corners[2, :]

    if H_est is None:
        return np.inf

    # Transform corners with estimated homography
    est_corners = H_est @ corners
    if np.any(np.abs(est_corners[2, :]) < 1e-7):
        return np.inf
        
    est_corners = est_corners[:2, :] / est_corners[2, :]
    dist = np.linalg.norm(gt_corners - est_corners, axis=0)
    return np.mean(dist)

def main():
    args = parse_args()
    top_k = args.top_k if args.matcher != 'xfeat-star' else 10000
    
    print(f"Loading XFeat model ({args.matcher}) with top_k={top_k}...")
    if args.use_original:
        xfeat = XFeat_original(weights=args.checkpoint, top_k=top_k)
    else: 
        xfeat = XFeat(weights=args.checkpoint, top_k=top_k, ablation_type=args.exp)
    matcher_fn = xfeat.match_xfeat_star if args.matcher == 'xfeat-star' else xfeat.match_xfeat

    seq_dirs = sorted(glob.glob(os.path.join(args.data_root, '*')))
    seq_dirs =[d for d in seq_dirs if os.path.isdir(d)]
    
    thresholds =[3, 5, 7]
    metrics = {
        'i': {thr:[] for thr in thresholds}, 
        'v': {thr:[] for thr in thresholds},
    }
    
    inlier_ratios = {'i': [], 'v':[]}
    inlier_counts = {'i': [], 'v': []}
    match_counts = {'i':[], 'v':[]}

    for seq_dir in tqdm(seq_dirs, desc="Evaluating HPatches"):
        seq_name = os.path.basename(seq_dir)
        seq_type = 'i' if seq_name.startswith('i_') else 'v'

        ref_img_path = os.path.join(seq_dir, '1.ppm')
        img0 = cv2.imread(ref_img_path)
        if img0 is None:
            continue
            
        img_shape = img0.shape[:2]

        for i in range(2, 7):
            tgt_img_path = os.path.join(seq_dir, f'{i}.ppm')
            img1 = cv2.imread(tgt_img_path)
            
            # matching points
            with torch.no_grad():
                mkpts0, mkpts1 = matcher_fn(img0, img1)

            if torch.is_tensor(mkpts0):
                mkpts0 = mkpts0.cpu().numpy()
            if torch.is_tensor(mkpts1):
                mkpts1 = mkpts1.cpu().numpy()

            # MAGSAC++
            H_est = None
            inlier_count = 0
            match_count = len(mkpts0)
            
            if match_count >= 4:
                H_est, inliers = cv2.findHomography(
                    mkpts0, mkpts1, cv2.USAC_MAGSAC, args.ransac_thr
                )
                if inliers is not None:
                    inlier_count = int(inliers.sum())

            H_gt_path = os.path.join(seq_dir, f'H_1_{i}')
            H_gt = np.loadtxt(H_gt_path)

            error = compute_homography_error(H_est, H_gt, img_shape)
            
            for thr in thresholds:
                metrics[seq_type][thr].append(1 if error <= thr else 0)
                
            inlier_ratio = inlier_count / match_count if match_count > 0 else 0.0
            inlier_ratios[seq_type].append(inlier_ratio)
            inlier_counts[seq_type].append(inlier_count)
            match_counts[seq_type].append(match_count)

    print("\n" + "="*100)
    print(f" {'Method: ' + args.matcher.upper():<35} | RANSAC Thr: {args.ransac_thr}")
    print("="*100)
    print(f" {'Sequence':<14} | {'MHA@3':<17} | {'MHA@5':<17} | {'MHA@7':<17} | {'MIR':<17}")
    print("-" * 100)

    def print_row(t_name, t_idx):
        d3 = metrics[t_idx][3]
        d5 = metrics[t_idx][5]
        d7 = metrics[t_idx][7]
        irs = inlier_ratios[t_idx]

        n = len(d3)
        if n > 1:
            m3, se3 = np.mean(d3)*100, np.std(d3, ddof=1)/np.sqrt(n)*100
            m5, se5 = np.mean(d5)*100, np.std(d5, ddof=1)/np.sqrt(n)*100
            m7, se7 = np.mean(d7)*100, np.std(d7, ddof=1)/np.sqrt(n)*100
            mir, mir_se = np.mean(irs)*100, np.std(irs, ddof=1)/np.sqrt(n)*100
            
            s3 = f"{m3:.2f}% ± {se3:.2f}%"
            s5 = f"{m5:.2f}% ± {se5:.2f}%"
            s7 = f"{m7:.2f}% ± {se7:.2f}%"
            smir = f"{mir:.2f}% ± {mir_se:.2f}%"
            
            print(f" {t_name:<14} | {s3:<17} | {s5:<17} | {s7:<17} | {smir:<17}")
        else:
            print(f" {t_name:<14} | Not enough data to compute SE.")

    print_row('Illumination', 'i')
    print_row('Viewpoint', 'v')
    print("="*96)
    
    all_inliers = inlier_counts['i'] + inlier_counts['v']
    all_matches = match_counts['i'] + match_counts['v']
    n_all = len(all_inliers)
    
    if n_all > 1:
        mean_matches = float(np.mean(all_matches))
        mean_matches_se = float(np.std(all_matches, ddof=1) / np.sqrt(n_all))
        
        mean_inliers = float(np.mean(all_inliers))
        mean_inliers_se = float(np.std(all_inliers, ddof=1) / np.sqrt(n_all))
        
        print(f"\nMean #Matches: {mean_matches:.2f} ± {mean_matches_se:.2f}")
        print(f"Mean #Inliers: {mean_inliers:.2f} ± {mean_inliers_se:.2f}\n")

if __name__ == '__main__':
    main()