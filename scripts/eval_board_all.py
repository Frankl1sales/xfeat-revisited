import glob
import json
import os
from pathlib import Path
import cv2
import numpy as np
import poselib
from tqdm import tqdm

W, H = 384.0, 288.0

def load_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 64), dtype=np.float32)
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    kpts = arr[:, :2]
    descs = arr[:, 2:66]
    descs = descs / (np.linalg.norm(descs, axis=1, keepdims=True) + 1e-8)
    return kpts, descs

def match_mnn(k0, d0, k1, d1):
    if len(k0) == 0 or len(k1) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    cossim = d0 @ d1.T
    m12 = np.argmax(cossim, axis=1)
    m21 = np.argmax(cossim, axis=0)
    idx0 = np.arange(len(m12))
    mutual = m21[m12] == idx0
    return k0[idx0[mutual]].copy(), k1[m12[mutual]].copy()

def intrinsics_to_camera(K):
    return {"model": "PINHOLE", "width": -1, "height": -1,
            "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]}

def relative_pose_error(T_0to1, R, t):
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    if n < 1e-10:
        t_err = 0.0
    else:
        t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
        t_err = min(t_err, 180.0 - t_err)
    R_gt = T_0to1[:3, :3]
    cos = np.clip((np.trace(R.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
    R_err = np.rad2deg(abs(np.arccos(cos)))
    return float(t_err), float(R_err)

def error_auc(errors, thresholds=(5, 10, 20)):
    n = len(errors)
    errors_sorted = [0.0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors_sorted)))
    res = {}
    for thr in thresholds:
        last_idx = np.searchsorted(errors_sorted, thr)
        y = recall[:last_idx] + [recall[last_idx - 1]] if last_idx > 0 else [0, 0]
        x = errors_sorted[:last_idx] + [thr] if last_idx > 0 else [0, thr]
        res[f"auc@{thr}"] = float(np.trapezoid(y, x) / thr)
        sr = float(np.mean(errors <= thr))
        res[f"auc@{thr}_se"] = float(np.sqrt(sr * (1 - sr) / n)) if n > 0 else 0.0
    return res

def eval_pose_pairs(pair_records, ransac_thr):
    errors, inlier_ratios, inlier_counts, match_counts = [], [], [], []
    for p in pair_records:
        pts0, pts1 = p["pts0"], p["pts1"]
        n_m = len(pts0)
        match_counts.append(n_m)
        if n_m < 5:
            errors.append(np.inf)
            inlier_counts.append(0)
            inlier_ratios.append(0.0)
            continue
        M, info = poselib.estimate_relative_pose(
            pts0.tolist(), pts1.tolist(),
            intrinsics_to_camera(p["K0"]), intrinsics_to_camera(p["K1"]),
            {"max_epipolar_error": ransac_thr, "success_prob": 0.99999,
             "min_iterations": 20, "max_iterations": 10000}
        )
        inl = np.array(info["inliers"])
        n_inl = int(np.count_nonzero(inl))
        t_err, R_err = relative_pose_error(p["T_0to1"], M.R, M.t)
        errors.append(max(t_err, R_err))
        inlier_counts.append(n_inl)
        inlier_ratios.append(n_inl / n_m if n_m > 0 else 0.0)

    errors = np.array(errors)
    n = len(errors)
    m = error_auc(errors, (5, 10, 20))
    acc10 = float((errors <= 10).mean())
    m["Acc@10"] = acc10
    m["Acc@10_se"] = float(np.sqrt(acc10 * (1 - acc10) / n))
    m["MIR"] = float(np.mean(inlier_ratios))
    m["MIR_se"] = float(np.std(inlier_ratios, ddof=1) / np.sqrt(n))
    m["mean_inliers"] = float(np.mean(inlier_counts))
    m["mean_inliers_se"] = float(np.std(inlier_counts, ddof=1) / np.sqrt(n))
    m["mean_matches"] = float(np.mean(match_counts))
    return m

def eval_megadepth_board():
    json_path = "assets/megadepth_1500_384x288.json"
    feat_root = Path("datasets/board_features/MegaDepth")
    if not feat_root.exists():
        return None
    with open(json_path) as f:
        pairs = json.load(f)
    cache = {}
    records = []
    for p in tqdm(pairs, desc="Board MegaDepth-1500"):
        n0, n1 = p["pair_names"]
        if n0 not in cache:
            cache[n0] = load_csv(feat_root / Path(n0).with_suffix(".csv"))
        if n1 not in cache:
            cache[n1] = load_csv(feat_root / Path(n1).with_suffix(".csv"))
        k0, d0 = cache[n0]
        k1, d1 = cache[n1]
        pts0, pts1 = match_mnn(k0, d0, k1, d1)
        if len(pts0) > 0:
            pts0 *= np.array(p["scale0"], dtype=np.float32)
            pts1 *= np.array(p["scale1"], dtype=np.float32)
        records.append({
            "pts0": pts0, "pts1": pts1,
            "K0": np.array(p["K0"], dtype=np.float32),
            "K1": np.array(p["K1"], dtype=np.float32),
            "T_0to1": np.array(p["T_0to1"], dtype=np.float32),
        })
    metrics = eval_pose_pairs(records, ransac_thr=2.5)
    out = {"dataset": "megadepth1500", "matcher": "rv1103_allkpts_384x288", "ransac_thr": 2.5, "num_pairs": len(records), "metrics": metrics}
    os.makedirs("output/megadepth_board_384x288", exist_ok=True)
    with open("output/megadepth_board_384x288/megadepth1500_board_ransac2.5.json", "w") as f:
        json.dump(out, f, indent=2)
    return out

def eval_hpatches_board():
    data_root = Path("datasets/hpatches-sequences-release")
    feat_root = Path("datasets/board_features/HPatches")
    if not feat_root.exists() or not data_root.exists():
        return None
    seq_dirs = sorted([d for d in data_root.glob("*") if d.is_dir()])
    thresholds = [3, 5, 7]
    metrics = {"i": {t: [] for t in thresholds}, "v": {t: [] for t in thresholds}}
    inlier_ratios = {"i": [], "v": []}
    inlier_counts = {"i": [], "v": []}
    match_counts = {"i": [], "v": []}

    for seq_dir in tqdm(seq_dirs, desc="Board HPatches"):
        seq_name = seq_dir.name
        seq_type = "i" if seq_name.startswith("i_") else "v"
        img0 = cv2.imread(str(seq_dir / "1.ppm"))
        if img0 is None:
            continue
        h0, w0 = img0.shape[:2]
        k0, d0 = load_csv(feat_root / seq_name / "1.csv")

        for i in range(2, 7):
            img1 = cv2.imread(str(seq_dir / f"{i}.ppm"))
            if img1 is None:
                continue
            h1, w1 = img1.shape[:2]
            k1, d1 = load_csv(feat_root / seq_name / f"{i}.csv")
            pts0, pts1 = match_mnn(k0, d0, k1, d1)
            if len(pts0) > 0:
                pts0[:, 0] *= (w0 / W); pts0[:, 1] *= (h0 / H)
                pts1[:, 0] *= (w1 / W); pts1[:, 1] *= (h1 / H)

            H_est, inl_cnt, m_cnt = None, 0, len(pts0)
            if m_cnt >= 4:
                H_est, inliers = cv2.findHomography(pts0, pts1, cv2.USAC_MAGSAC, 1.5)
                if inliers is not None:
                    inl_cnt = int(inliers.sum())

            H_gt = np.loadtxt(str(seq_dir / f"H_1_{i}"))
            corners = np.array([[0, 0, 1], [w0 - 1, 0, 1], [0, h0 - 1, 1], [w0 - 1, h0 - 1, 1]]).T
            gt_c = H_gt @ corners
            gt_c = gt_c[:2] / gt_c[2]
            if H_est is None:
                err = np.inf
            else:
                est_c = H_est @ corners
                err = np.inf if np.any(np.abs(est_c[2]) < 1e-7) else float(np.mean(np.linalg.norm(gt_c - (est_c[:2] / est_c[2]), axis=0)))

            for t in thresholds:
                metrics[seq_type][t].append(1 if err <= t else 0)
            inlier_ratios[seq_type].append(inl_cnt / m_cnt if m_cnt > 0 else 0.0)
            inlier_counts[seq_type].append(inl_cnt)
            match_counts[seq_type].append(m_cnt)

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

    out = {
        "dataset": "hpatches",
        "matcher": "rv1103_allkpts_384x288",
        "ransac_thr": 1.5,
        "illumination": summarize("i"),
        "viewpoint": summarize("v"),
        "mean_matches": float(np.mean(match_counts["i"] + match_counts["v"])),
        "mean_inliers": float(np.mean(inlier_counts["i"] + inlier_counts["v"])),
    }
    os.makedirs("output/hpatches_board_384x288", exist_ok=True)
    with open("output/hpatches_board_384x288/hpatches_board_ransac1.5.json", "w") as f:
        json.dump(out, f, indent=2)
    return out

def eval_scannet_board():
    data_root = Path("datasets/ScanNet1500/scannet_test_1500")
    npz_path = Path("datasets/ScanNet1500/test.npz")
    feat_root = Path("datasets/board_features/ScanNet")
    if not feat_root.exists() or not npz_path.exists():
        return None
    names = np.load(npz_path)["name"]
    cache, k_cache, records = {}, {}, []
    for i in tqdm(range(len(names)), desc="Board ScanNet-1500"):
        scene = f"scene{int(names[i,0]):04d}_{int(names[i,1]):02d}"
        im0, im1 = str(int(names[i, 2])), str(int(names[i, 3]))
        if scene not in k_cache:
            k_cache[scene] = np.loadtxt(data_root / scene / "intrinsic/intrinsic_color.txt").astype(np.float32)
        K = k_cache[scene]
        p0 = np.loadtxt(data_root / scene / "pose" / f"{im0}.txt")
        p1 = np.loadtxt(data_root / scene / "pose" / f"{im1}.txt")
        R_0to1 = p1[:3, :3].T @ p0[:3, :3]
        t_0to1 = p1[:3, :3].T @ (p0[:3, [3]] - p1[:3, [3]])
        T_0to1 = np.concatenate([R_0to1, t_0to1], axis=-1).astype(np.float32)

        rel0, rel1 = f"{scene}/color/{im0}.csv", f"{scene}/color/{im1}.csv"
        if rel0 not in cache: cache[rel0] = load_csv(feat_root / rel0)
        if rel1 not in cache: cache[rel1] = load_csv(feat_root / rel1)
        k0, d0 = cache[rel0]
        k1, d1 = cache[rel1]
        pts0, pts1 = match_mnn(k0, d0, k1, d1)
        if len(pts0) > 0:
            # ScanNet color images are 640x480
            pts0[:, 0] *= (640.0 / W); pts0[:, 1] *= (480.0 / H)
            pts1[:, 0] *= (640.0 / W); pts1[:, 1] *= (480.0 / H)
        records.append({"pts0": pts0, "pts1": pts1, "K0": K, "K1": K, "T_0to1": T_0to1})

    m05 = eval_pose_pairs(records, ransac_thr=0.5)
    m25 = eval_pose_pairs(records, ransac_thr=2.5)
    out = {"dataset": "scannet1500", "matcher": "rv1103_allkpts_384x288",
           "num_pairs": len(records), "metrics_ransac0.5": m05, "metrics_ransac2.5": m25}
    os.makedirs("output/scannet_board_384x288", exist_ok=True)
    with open("output/scannet_board_384x288/scannet1500_board.json", "w") as f:
        json.dump(out, f, indent=2)
    return out

def main():
    md = eval_megadepth_board()
    hp = eval_hpatches_board()
    sn = eval_scannet_board()

    def load_j(p):
        return json.load(open(p)) if os.path.exists(p) else None

    md_384 = load_j("output/megadepth_fp32_384x288/megadepth1500_xfeat_ransac2.5.json")
    md_ori = load_j("output/megadepth_fp32_orig/megadepth1500_xfeat_ransac2.5.json")
    hp_384 = load_j("output/hpatches_fp32_384x288/hpatches_fp32.json")
    hp_ori = load_j("output/hpatches_fp32_orig/hpatches_fp32.json")
    sn_384 = load_j("output/scannet_fp32_384x288/scannet1500_xfeat_ransac0.5.json")
    sn_ori = load_j("output/scannet_fp32_orig/scannet1500_xfeat_ransac0.5.json")

    lines = ["# Benchmark XFeat: Luckfox RV1103 (INT8 Split) vs PyTorch FP32\n"]
    lines.append("## 1. MegaDepth-1500 (Outdoor Relative Pose, RANSAC 2.5px)\n")
    lines.append("| Configuração | AUC@5 | AUC@10 | AUC@20 | Acc@10° | MIR | Mean Inliers |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, obj in [("PyTorch FP32 Original (top_k=4096)", md_ori),
                       ("PyTorch FP32 384x288 (Baseline A)", md_384),
                       ("Luckfox RV1103 INT8 384x288 (allkpts)", md)]:
        if obj and "metrics" in obj:
            m = obj["metrics"]
            lines.append(f"| {label} | {m['auc@5']*100:.2f}% | {m['auc@10']*100:.2f}% | {m['auc@20']*100:.2f}% | {m['Acc@10']*100:.2f}% | {m['MIR']*100:.2f}% | {m['mean_inliers']:.1f} |")

    lines.append("\n## 2. HPatches (Homography Estimation, MAGSAC 1.5px)\n")
    lines.append("| Configuração | Illum MHA@3 | Illum MHA@5 | Illum MHA@7 | View MHA@3 | View MHA@5 | View MHA@7 | Mean Inliers |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, obj in [("PyTorch FP32 Original (top_k=4096)", hp_ori),
                       ("PyTorch FP32 384x288 (Baseline A)", hp_384),
                       ("Luckfox RV1103 INT8 384x288 (allkpts)", hp)]:
        if obj and "illumination" in obj:
            i, v = obj["illumination"], obj["viewpoint"]
            lines.append(f"| {label} | {i['MHA@3']*100:.2f}% | {i['MHA@5']*100:.2f}% | {i['MHA@7']*100:.2f}% | {v['MHA@3']*100:.2f}% | {v['MHA@5']*100:.2f}% | {v['MHA@7']*100:.2f}% | {obj['mean_inliers']:.1f} |")

    lines.append("\n## 3. ScanNet-1500 (Indoor Relative Pose, RANSAC 0.5px)\n")
    lines.append("| Configuração | AUC@5 | AUC@10 | AUC@20 |")
    lines.append("|---|---|---|---|")
    for label, obj, key in [("PyTorch FP32 Original (top_k=4096)", sn_ori, "metrics"),
                            ("PyTorch FP32 384x288 (Baseline A)", sn_384, "metrics"),
                            ("Luckfox RV1103 INT8 384x288 (allkpts)", sn, "metrics_ransac0.5")]:
        if obj and key in obj:
            m = obj[key]
            lines.append(f"| {label} | {m['auc@5']*100:.2f}% | {m['auc@10']*100:.2f}% | {m['auc@20']*100:.2f}% |")

    report = "\n".join(lines) + "\n"
    with open("output/BENCHMARK_SUMMARY_REPORT.md", "w") as f:
        f.write(report)
    print("\n" + report)

if __name__ == "__main__":
    main()
