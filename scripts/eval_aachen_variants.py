import argparse
import csv
import shutil
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import hloc
from hloc import extract_features, match_features, triangulation, localize_sfm
from hloc.utils.base_model import BaseModel

class XFeatResizedExtractor(BaseModel):
    default_conf = {'max_keypoints': 4096, 'weights': '', 'resize_hw': [288, 384]}
    required_inputs = ['image']

    def _init(self, conf):
        from modules.xfeat_original import XFeat as XFeatOriginal
        self.xfeat = XFeatOriginal(weights=conf['weights'], top_k=conf['max_keypoints'])

    def _forward(self, data):
        img = data['image']  # (1, C, H, W)
        _, _, h_orig, w_orig = img.shape
        h_t, w_t = self.conf['resize_hw']
        img_resized = F.interpolate(img, size=(h_t, w_t), mode='bilinear', align_corners=False)
        out = self.xfeat.detectAndCompute(img_resized, top_k=self.conf['max_keypoints'])[0]
        kpts = out['keypoints'].clone()
        kpts[:, 0] *= (w_orig / float(w_t))
        kpts[:, 1] *= (h_orig / float(h_t))
        scores = out.get('scores', torch.ones(len(kpts), device=kpts.device))
        return {
            'keypoints': kpts.unsqueeze(0),
            'descriptors': out['descriptors'].T.unsqueeze(0),
            'scores': scores.unsqueeze(0)
        }

_orig_load = hloc.utils.base_model.dynamic_load
hloc.utils.base_model.dynamic_load = (
    lambda module, name: XFeatResizedExtractor if name == 'xfeat_resized' else _orig_load(module, name)
)
hloc.extract_features.dynamic_load = hloc.utils.base_model.dynamic_load

def build_h5_from_board_csvs(images_dir: Path, csv_root: Path, h5_path: Path):
    import cv2
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    img_files = sorted([p for p in images_dir.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    with h5py.File(str(h5_path), 'w') as fd:
        for img_path in img_files:
            rel = img_path.relative_to(images_dir).as_posix()
            csv_file = csv_root / (rel + '.csv')
            im = cv2.imread(str(img_path))
            h_orig, w_orig = im.shape[:2]
            sx, sy = w_orig / 384.0, h_orig / 288.0
            kpts, scores, descs = [], [], []
            if csv_file.exists():
                with open(csv_file, 'r') as f:
                    for row in csv.reader(f):
                        if not row or row[0] == 'x':
                            continue
                        kpts.append([float(row[0]) * sx, float(row[1]) * sy])
                        scores.append(float(row[2]))
                        descs.append([float(v) for v in row[3:67]])
            grp = fd.create_group(rel)
            if kpts:
                grp.create_dataset('keypoints', data=np.array(kpts, dtype=np.float32))
                grp.create_dataset('scores', data=np.array(scores, dtype=np.float32))
                grp.create_dataset('descriptors', data=np.array(descs, dtype=np.float32).T)  # (64, N)
            else:
                grp.create_dataset('keypoints', data=np.zeros((0, 2), dtype=np.float32))
                grp.create_dataset('scores', data=np.zeros((0,), dtype=np.float32))
                grp.create_dataset('descriptors', data=np.zeros((64, 0), dtype=np.float32))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fp32_384x288', 'board_int8'], required=True)
    ap.add_argument('--data-root', default='datasets/aachen')
    ap.add_argument('--ref-outputs', default='output/aachen_fp32_orig')
    ap.add_argument('--outputs-dir', required=True)
    ap.add_argument('--checkpoint', default='checkpoints/xfeat.pt')
    ap.add_argument('--board-csv-root', default='datasets/board_features/aachen')
    args = ap.parse_args()

    dataset = Path(args.data_root)
    ref_out = Path(args.ref_outputs)
    outputs = Path(args.outputs_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    images = dataset / 'images/images_upright'

    sift_sfm = ref_out / 'sfm_sift'
    sfm_pairs = ref_out / 'pairs-db-covis20.txt'
    loc_pairs = ref_out / 'pairs-query-netvlad50.txt'
    combined_queries = ref_out / 'aachen_v1.0_combined_queries.txt'

    if args.mode == 'fp32_384x288':
        xfeat_conf = {
            'output': 'feats',
            'model': {'name': 'xfeat_resized', 'max_keypoints': 4096, 'weights': args.checkpoint, 'resize_hw': [288, 384]},
            'preprocessing': {'grayscale': False, 'resize_max': 1024}
        }
        feature_path = extract_features.main(xfeat_conf, images, outputs)
    else:
        feature_path = outputs / 'feats.h5'
        build_h5_from_board_csvs(images, Path(args.board_csv_root), feature_path)

    match_conf = {'output': 'matches', 'model': {'name': 'nearest_neighbor', 'mutual_check': True}}
    match_path = match_features.main(match_conf, sfm_pairs, feature_path, outputs)
    reference_sfm = outputs / 'sfm'
    triangulation.main(reference_sfm, sift_sfm, images, sfm_pairs, feature_path, match_path)
    match_features.main(match_conf, loc_pairs, feature_path, outputs, match_path)
    results_txt = outputs / 'Aachen_v1.0_hloc.txt'
    localize_sfm.main(reference_sfm, combined_queries, loc_pairs, feature_path, match_path, results_txt)
    print(f"[OK] Aachen ({args.mode}) salvo em: {results_txt}")

if __name__ == '__main__':
    main()
