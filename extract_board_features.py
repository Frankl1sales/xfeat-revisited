import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np

REPO = Path("/media/frank/01DBA5D4E1C670D06/Feature_Matching/xfeat-revisited")
DATASET = REPO / "datasets/Mega1500/megadepth_test_1500"
PAIRS_JSON = REPO / "assets/megadepth_1500_384x288.json"
OUT_ROOT = REPO / "datasets/board_features/MegaDepth"

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
            "-p", BOARD_PORT, f"{BOARD_USER}@{BOARD_HOST}", cmd]
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout).stdout

def scp_put(local, remote):
    subprocess.run(["sshpass", "-p", BOARD_PASS, "scp", "-q",
                    "-o", "StrictHostKeyChecking=no",
                    "-P", BOARD_PORT, str(local),
                    f"{BOARD_USER}@{BOARD_HOST}:{remote}"], capture_output=True)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Processar apenas N imagens para teste")
    args = ap.parse_args()

    with open(PAIRS_JSON) as f:
        pairs = json.load(f)

    images = sorted({name for p in pairs for name in p["pair_names"]})
    if args.limit > 0:
        images = images[:args.limit]
    print(f"Imagens a processar: {len(images)} | Saída: {OUT_ROOT}")

    n_done = n_skip = n_fail = 0
    t0 = time.time()

    for i, name in enumerate(images):
        out_csv = OUT_ROOT / Path(name).with_suffix(".csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        if out_csv.exists() and out_csv.stat().st_size > 0:
            n_skip += 1
            continue

        img_path = DATASET / name
        if not img_path.exists():
            print(f"[fail] não existe: {img_path}")
            n_fail += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[fail] erro ao ler: {img_path}")
            n_fail += 1
            continue

        img_small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_small)

        try:
            kpts, descs = run_board_allkpts(tmp_path, f"md_{i:05d}.png")
            with open(out_csv, "w") as f:
                for kp, d in zip(kpts, descs):
                    f.write(f"{kp[0]:.3f},{kp[1]:.3f}," + ",".join(f"{v:.6f}" for v in d) + "\n")
            n_done += 1
            if n_done <= 5 or n_done % 25 == 0:
                dt = time.time() - t0
                print(f"[{i+1}/{len(images)}] {name} -> {len(kpts)} kpts ({dt:.1f}s)")
        except Exception as e:
            print(f"[fail] {name}: {e}")
            n_fail += 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    print(f"\n=== CONCLUÍDO === done={n_done} skip={n_skip} fail={n_fail}")

if __name__ == "__main__":
    main()
