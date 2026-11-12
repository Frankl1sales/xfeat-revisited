#!/usr/bin/env bash
set -u
BASE="/media/frank/01DBA5D4E1C670D06/Feature_Matching"
REPO="$BASE/xfeat-revisited"
BRIDGE="$BASE/luckfox-xfeat-quantization/_docker_bridge_scannet_cpu"
LOG="$REPO/output/scannet_fix.log"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date +%H:%M:%S)] Rodando ScanNet FP32 384x288 (Baseline A real) na CPU..."

docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_scannet_cpu
mkdir -p "$BRIDGE/datasets" "$BRIDGE/output"
cp -r "$REPO/modules" "$REPO/assets" "$REPO/checkpoints" "$REPO/scripts" "$BRIDGE/"
cp -al "$REPO/datasets/ScanNet1500" "$BRIDGE/datasets/"
cp -r "$REPO/output/"* "$BRIDGE/output/"

docker exec -w /workspace/_docker_bridge_scannet_cpu xfeat_exportador \
  /usr/bin/python3 - << 'PYEOF'
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from modules.xfeat_original import XFeat as XFeatOriginal
from modules.evaluation.eval import compute_pose_error, pose_auc

repo = Path("/workspace/_docker_bridge_scannet_cpu")
npz = np.load(repo / "datasets/ScanNet1500/test.npz")
names, intrinsics, rel_poses = npz["name"], npz["intrinsics"], npz["rel_poses"]

xfeat = XFeatOriginal(weights=str(repo / "checkpoints/xfeat.pt"), top_k=4096)
errs = []

for i in tqdm(range(len(names)), desc="ScanNet FP32 384x288"):
    scene = f"scene{int(names[i,0]):04d}_{int(names[i,1]):02d}"
    p0 = repo / f"datasets/ScanNet1500/scannet_test_1500/{scene}/color/{int(names[i,2])}.jpg"
    p1 = repo / f"datasets/ScanNet1500/scannet_test_1500/{scene}/color/{int(names[i,3])}.jpg"
    im0, im1 = cv2.imread(str(p0)), cv2.imread(str(p1))
    h0, w0 = im0.shape[:2]
    h1, w1 = im1.shape[:2]

    im0_r = cv2.resize(im0, (384, 288), interpolation=cv2.INTER_AREA)
    im1_r = cv2.resize(im1, (384, 288), interpolation=cv2.INTER_AREA)
    out0 = xfeat.detectAndCompute(im0_r, top_k=4096)[0]
    out1 = xfeat.detectAndCompute(im1_r, top_k=4096)[0]

    idxs0, idxs1 = xfeat.match(out0["descriptors"], out1["descriptors"])
    mkpts0 = out0["keypoints"][idxs0].cpu().numpy().copy()
    mkpts1 = out1["keypoints"][idxs1].cpu().numpy().copy()
    mkpts0[:, 0] *= (w0 / 384.0); mkpts0[:, 1] *= (h0 / 288.0)
    mkpts1[:, 0] *= (w1 / 384.0); mkpts1[:, 1] *= (h1 / 288.0)

    K0, K1, T_0to1 = intrinsics[i, 0], intrinsics[i, 1], rel_poses[i]
    if len(mkpts0) < 5:
        errs.append(180.0); continue
    k0n = (mkpts0 - K0[[0, 1], [2, 2]]) / K0[[0, 1], [0, 1]]
    k1n = (mkpts1 - K1[[0, 1], [2, 2]]) / K1[[0, 1], [0, 1]]
    f_mean = 0.5 * (K0[0, 0] + K1[0, 0])
    E, mask = cv2.findEssentialMat(k0n, k1n, np.eye(3), method=cv2.RANSAC, prob=0.99999, threshold=0.5 / f_mean)
    if E is None or mask is None or mask.ravel().sum() < 5:
        errs.append(180.0); continue
    mask = mask.ravel().astype(bool)
    _, R, t, _ = cv2.recoverPose(E, k0n[mask], k1n[mask], np.eye(3))
    t_err, r_err = compute_pose_error(T_0to1, R, t.ravel())
    errs.append(max(t_err, r_err))

aucs = pose_auc(np.array(errs), thresholds=[5, 10, 20])
out_dir = repo / "output/scannet_fp32_384x288"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "scannet1500_xfeat_ransac0.5.json", "w") as f:
    json.dump({"metrics": {"auc@5": float(aucs[0]), "auc@10": float(aucs[1]), "auc@20": float(aucs[2])}}, f, indent=2)
print(f"ScanNet FP32 384x288 -> AUC@5={aucs[0]*100:.2f}% | AUC@10={aucs[1]*100:.2f}% | AUC@20={aucs[2]*100:.2f}%")
PYEOF

docker exec xfeat_exportador chmod -R a+rwX /workspace/_docker_bridge_scannet_cpu/output
cp -r "$BRIDGE/output/"* "$REPO/output/"
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_scannet_cpu
echo "[$(date +%H:%M:%S)] ScanNet FP32 384x288 concluído!"
