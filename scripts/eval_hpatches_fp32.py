import sys, os; sys.path.insert(0, os.getcwd())
import argparse
import glob
import json
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from modules.xfeat import XFeat
from modules.xfeat_original import XFeat as XFeat_original
from modules.evaluation.eval_hpatches import compute_homography_error

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets/hpatches-sequences-release")
    ap.add_argument("--checkpoint", default="checkpoints/xfeat.pt")
    ap.add_argument("--use-original", action="store_true")
    ap.add_argument("--top-k", type=int, default=4096)
    ap.add_argument("--ransac-thr", type=float, default=1.5)
    ap.add_argument("--resize-hw", type=int, nargs=2, default=None)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    if args.use_original:
        xfeat = XFeat_original(weights=args.checkpoint, top_k=args.top_k)
    else:
        xfeat = XFeat(weights=args.checkpoint, top_k=args.top_k)

    seq_dirs = sorted([d for d in glob.glob(os.path.join(args.data_root, "*")) if os.path.isdir(d)])
    thresholds = [3, 5, 7]
    metrics = {"i": {t: [] for t in thresholds}, "v": {t: [] for t in thresholds}}
    inlier_ratios = {"i": [], "v": []}
    inlier_counts = {"i": [], "v": []}
    match_counts = {"i": [], "v": []}

    for seq_dir in tqdm(seq_dirs, desc=f"HPatches ({args.resize_hw or 'native'})"):
        seq_name = os.path.basename(seq_dir)
        seq_type = "i" if seq_name.startswith("i_") else "v"
        img0 = cv2.imread(os.path.join(seq_dir, "1.ppm"))
        if img0 is None:
            continue
        h0, w0 = img0.shape[:2]
        in0 = cv2.resize(img0, (args.resize_hw[1], args.resize_hw[0]), interpolation=cv2.INTER_AREA) if args.resize_hw else img0

        for i in range(2, 7):
            img1 = cv2.imread(os.path.join(seq_dir, f"{i}.ppm"))
            if img1 is None:
                continue
            h1, w1 = img1.shape[:2]
            in1 = cv2.resize(img1, (args.resize_hw[1], args.resize_hw[0]), interpolation=cv2.INTER_AREA) if args.resize_hw else img1

            with torch.no_grad():
                mkpts0, mkpts1 = xfeat.match_xfeat(in0, in1)
            if torch.is_tensor(mkpts0): mkpts0 = mkpts0.cpu().numpy()
            if torch.is_tensor(mkpts1): mkpts1 = mkpts1.cpu().numpy()

            if args.resize_hw and len(mkpts0) > 0:
                rh, rw = args.resize_hw
                mkpts0[:, 0] *= (w0 / float(rw))
                mkpts0[:, 1] *= (h0 / float(rh))
                mkpts1[:, 0] *= (w1 / float(rw))
                mkpts1[:, 1] *= (h1 / float(rh))

            H_est, inlier_count, match_count = None, 0, len(mkpts0)
            if match_count >= 4:
                H_est, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, args.ransac_thr)
                if inliers is not None:
                    inlier_count = int(inliers.sum())

            H_gt = np.loadtxt(os.path.join(seq_dir, f"H_1_{i}"))
            err = compute_homography_error(H_est, H_gt, (h0, w0))
            for t in thresholds:
                metrics[seq_type][t].append(1 if err <= t else 0)
            inlier_ratios[seq_type].append(inlier_count / match_count if match_count > 0 else 0.0)
            inlier_counts[seq_type].append(inlier_count)
            match_counts[seq_type].append(match_count)

    def summarize(k):
        n = len(metrics[k][3])
        res = {"num_pairs": n}
        for t in thresholds:
            arr = np.array(metrics[k][t], dtype=float)
            res[f"MHA@{t}"] = float(arr.mean())
            res[f"MHA@{t}_se"] = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        ir = np.array(inlier_ratios[k], dtype=float)
        res["MIR"] = float(ir.mean())
        res["MIR_se"] = float(ir.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return res

    all_m = match_counts["i"] + match_counts["v"]
    all_i = inlier_counts["i"] + inlier_counts["v"]
    out = {
        "dataset": "hpatches",
        "resize_hw": args.resize_hw,
        "top_k": args.top_k,
        "ransac_thr": args.ransac_thr,
        "illumination": summarize("i"),
        "viewpoint": summarize("v"),
        "mean_matches": float(np.mean(all_m)),
        "mean_inliers": float(np.mean(all_i)),
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
