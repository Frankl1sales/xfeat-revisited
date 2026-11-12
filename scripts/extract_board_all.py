import argparse
import glob
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np

REPO = Path("/media/frank/01DBA5D4E1C670D06/Feature_Matching/xfeat-revisited")
BOARD_HOST = "localhost"
BOARD_PORT = "2222"
BOARD_USER = "root"
BOARD_PASS = os.environ.get("BOARD_PASS", "luckfox")
W, H = 384, 288
BIN_NAME = "./luckfox_pico_xfeat_allkpts"
RKNN_A = f"xfeat_A_{W}x{H}.rknn"
RKNN_B = f"xfeat_B_{W}x{H}.rknn"

def ssh(cmd, timeout=120):
    full = ["sshpass", "-p", BOARD_PASS, "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=15", "-p", BOARD_PORT, f"{BOARD_USER}@{BOARD_HOST}", cmd]
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout).stdout

def scp_put(local, remote):
    subprocess.run(["sshpass", "-p", BOARD_PASS, "scp", "-q",
                    "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
                    "-P", BOARD_PORT, str(local), f"{BOARD_USER}@{BOARD_HOST}:{remote}"],
                   capture_output=True, timeout=60)

def run_board_allkpts(img_path, remote_name):
    scp_put(img_path, f"/root/deploy_v2/{remote_name}")
    out = ssh(f"cd /root/deploy_v2 && LD_LIBRARY_PATH=lib {BIN_NAME} {RKNN_A} {RKNN_B} {remote_name} {W} {H} 2>&1; rm -f /root/deploy_v2/{remote_name}")
    kpts, descs = [], []
    inside = False
    for line in out.splitlines():
        s = line.strip()
        if s == "CSV_START":
            inside = True
            continue
        if s == "CSV_END":
            break
        if not inside or not s:
            continue
        p = s.split(",")
        if len(p) < 66:
            continue
        kpts.append((float(p[0]), float(p[1])))
        descs.append(np.array([float(x) for x in p[2:66]], dtype=np.float32))
    if not kpts:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 64), dtype=np.float32)
    return np.array(kpts, dtype=np.float32), np.array(descs, dtype=np.float32)

def collect_items(dataset):
    items = []
    if dataset == "megadepth":
        root = REPO / "datasets/Mega1500/megadepth_test_1500"
        out_root = REPO / "datasets/board_features/MegaDepth"
        with open(REPO / "assets/megadepth_1500_384x288.json") as f:
            pairs = json.load(f)
        for rel in sorted({n for p in pairs for n in p["pair_names"]}):
            items.append((root / rel, out_root / Path(rel).with_suffix(".csv")))
    elif dataset == "hpatches":
        root = REPO / "datasets/hpatches-sequences-release"
        out_root = REPO / "datasets/board_features/HPatches"
        for img_path in sorted(root.glob("*/*.ppm")):
            rel = img_path.relative_to(root)
            items.append((img_path, out_root / rel.with_suffix(".csv")))
    elif dataset == "scannet":
        root = REPO / "datasets/ScanNet1500/scannet_test_1500"
        npz = np.load(REPO / "datasets/ScanNet1500/test.npz")
        names = npz["name"]
        out_root = REPO / "datasets/board_features/ScanNet"
        rels = set()
        for i in range(len(names)):
            scene = f"scene{int(names[i,0]):04d}_{int(names[i,1]):02d}"
            rels.add(f"{scene}/color/{int(names[i,2])}.jpg")
            rels.add(f"{scene}/color/{int(names[i,3])}.jpg")
        for rel in sorted(rels):
            items.append((root / rel, out_root / Path(rel).with_suffix(".csv")))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["megadepth", "hpatches", "scannet"])
    args = ap.parse_args()

    items = collect_items(args.dataset)
    print(f"[{args.dataset.upper()}] Total de imagens únicas: {len(items)}")
    n_done = n_skip = n_fail = 0
    t0 = time.time()

    for i, (img_path, out_csv) in enumerate(items):
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if out_csv.exists() and out_csv.stat().st_size > 0:
            n_skip += 1
            continue
        if not img_path.exists():
            n_fail += 1
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            n_fail += 1
            continue

        img_small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_small)

        success = False
        for attempt in range(3):
            try:
                kpts, descs = run_board_allkpts(tmp_path, f"bm_{args.dataset[:2]}_{i:05d}.png")
                if len(kpts) > 0:
                    with open(out_csv, "w") as f:
                        for kp, d in zip(kpts, descs):
                            f.write(f"{kp[0]:.3f},{kp[1]:.3f}," + ",".join(f"{v:.6f}" for v in d) + "\n")
                    success = True
                    break
            except Exception as e:
                time.sleep(2)
        Path(tmp_path).unlink(missing_ok=True)

        if success:
            n_done += 1
            if n_done <= 3 or n_done % 50 == 0:
                print(f"[{args.dataset} {i+1}/{len(items)}] -> {len(kpts)} kpts ({time.time()-t0:.1f}s)")
        else:
            n_fail += 1

    print(f"[{args.dataset.upper()}] CONCLUÍDO: done={n_done} skip={n_skip} fail={n_fail}")

if __name__ == "__main__":
    main()
