import json
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm
from modules.xfeat_original import XFeat as XFeatOriginal
from modules.evaluation.eval import compute_pose_error, pose_auc

repo = Path("/workspace/_docker_bridge_scannet_cpu")
npz = np.load(repo / "datasets/ScanNet1500/test.npz")
names, intrinsics, rel_poses = npz["name"], npz["intrinsics"], npz["rel_poses"]
xfeat = XFeatOriginal(weights=str(repo / "checkpoints/xfeat.pt"), top_k=4096)
errs = []
for i in tqdm(range(len(names)), desc="ScanNet FP32 384x288"):
    sc = f"scene{int(names[i,0]):04d}_{int(names[i,1]):02d}"
    im0 = cv2.imread(str(repo / f"datasets/ScanNet1500/scannet_test_1500/{sc}/color/{int(names[i,2])}.jpg"))
    im1 = cv2.imread(str(repo / f"datasets/ScanNet1500/scannet_test_1500/{sc}/color/{int(names[i,3])}.jpg"))
    h0, w0 = im0.shape[:2]; h1, w1 = im1.shape[:2]
    o0 = xfeat.detectAndCompute(cv2.resize(im0, (384, 288), interpolation=cv2.INTER_AREA), top_k=4096)[0]
    o1 = xfeat.detectAndCompute(cv2.resize(im1, (384, 288), interpolation=cv2.INTER_AREA), top_k=4096)[0]
    i0, i1 = xfeat.match(o0["descriptors"], o1["descriptors"])
    k0 = o0["keypoints"][i0].cpu().numpy().copy(); k1 = o1["keypoints"][i1].cpu().numpy().copy()
    k0[:, 0] *= (w0 / 384.0); k0[:, 1] *= (h0 / 288.0)
    k1[:, 0] *= (w1 / 384.0); k1[:, 1] *= (h1 / 288.0)
    K0, K1 = intrinsics[i, 0], intrinsics[i, 1]
    if len(k0) < 5: errs.append(180.0); continue
    k0n = (k0 - K0[[0, 1], [2, 2]]) / K0[[0, 1], [0, 1]]
    k1n = (k1 - K1[[0, 1], [2, 2]]) / K1[[0, 1], [0, 1]]
    E, m = cv2.findEssentialMat(k0n, k1n, np.eye(3), method=cv2.RANSAC, prob=0.99999, threshold=0.5 / (0.5 * (K0[0, 0] + K1[0, 0])))
    if E is None or m is None or m.ravel().sum() < 5: errs.append(180.0); continue
    m = m.ravel().astype(bool)
    _, R, t, _ = cv2.recoverPose(E, k0n[m], k1n[m], np.eye(3))
    te, re = compute_pose_error(rel_poses[i], R, t.ravel())
    errs.append(max(te, re))
aucs = pose_auc(np.array(errs), thresholds=[5, 10, 20])
out = repo / "output/scannet_fp32_384x288"; out.mkdir(parents=True, exist_ok=True)
json.dump({"metrics": {"auc@5": float(aucs[0]), "auc@10": float(aucs[1]), "auc@20": float(aucs[2])}}, open(out / "scannet1500_xfeat_ransac0.5.json", "w"), indent=2)
print(f"ScanNet FP32 384x288 -> AUC@5={aucs[0]*100:.2f}% | AUC@10={aucs[1]*100:.2f}% | AUC@20={aucs[2]*100:.2f}%")
