import argparse
import os
import shutil
from pathlib import Path
import numpy as np
import cv2
import torch

import hloc
from hloc import extract_features, match_features, triangulation, localize_sfm
from hloc import pairs_from_covisibility, pairs_from_retrieval
from hloc.utils.base_model import BaseModel

class CustomXFeatExtractor(BaseModel):
    default_conf = {
        'max_keypoints': 4096,
        'weights': '',
        'use_original': False,
        'exp': 'skip_default'
    }
    required_inputs = ['image']

    def _init(self, conf):
        if conf['use_original']:
            from modules.xfeat_original import XFeat as XFeat_original
            self.xfeat = XFeat_original(weights=conf['weights'], top_k=conf['max_keypoints'])
        else:
            from modules.xfeat import XFeat
            self.xfeat = XFeat(weights=conf['weights'], top_k=conf['max_keypoints'], ablation_type=conf['exp'])
            
    def _forward(self, data):
        out = self.xfeat.detectAndCompute(data['image'], top_k=self.conf['max_keypoints'])[0]
        scores = out.get('scores', torch.ones(len(out['keypoints']), device=out['keypoints'].device))
        
        return {
            'keypoints': out['keypoints'].unsqueeze(0),       # (1, N, 2)
            'descriptors': out['descriptors'].T.unsqueeze(0), # (1, 64, N)
            'scores': scores.unsqueeze(0)                     # (1, N)
        }

_orig_load = hloc.utils.base_model.dynamic_load
hloc.utils.base_model.dynamic_load = lambda module, name: CustomXFeatExtractor if name == 'custom_xfeat' else _orig_load(module, name)
hloc.extract_features.dynamic_load = hloc.utils.base_model.dynamic_load

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XFeat on Aachen Day-Night")
    parser.add_argument('--data-root', type=str, help='Path to Aachen dataset folder', default='datasets/aachen')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint')
    parser.add_argument('--top-k', type=int, default=4096)
    parser.add_argument('--exp', type=str, default='skip_default')
    parser.add_argument('--use-original', action='store_true')
    parser.add_argument('--outputs-dir', type=str, default='outputs/aachen_xfeat')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print(f" Loading XFeat model | Original: {args.use_original}")
    print("="*80)

    dataset = Path(args.data_root)
    outputs = Path(args.outputs_dir)
    outputs.mkdir(exist_ok=True, parents=True)
    images = dataset / 'images/images_upright'

    def is_valid_colmap_model(model_dir: Path) -> bool:
        return all((model_dir / f).exists() for f in ('cameras.bin', 'images.bin', 'points3D.bin'))
    
    version = "v1.0"
    sift_sfm = outputs / 'sfm_sift'
    if not sift_sfm.exists() or not is_valid_colmap_model(sift_sfm):
        print("\n--- Converting original Aachen NVM model to COLMAP format ---")
        from hloc import colmap_from_nvm
        nvm = dataset / '3D-models/aachen_cvpr2018_db.nvm'
        intrinsics = dataset / '3D-models/database_intrinsics.txt'
        db_path = dataset / 'aachen.db'
        if not db_path.exists():
            raise FileNotFoundError(
                f"Missing {db_path}. Run python -m modules.datasets.download --aachen first."
            )

        # Clean partial/invalid model folders from interrupted previous runs.
        if sift_sfm.exists() and not is_valid_colmap_model(sift_sfm):
            print(f"Found incomplete COLMAP model at {sift_sfm}, regenerating...")
            shutil.rmtree(sift_sfm)

        colmap_from_nvm.main(nvm, intrinsics, db_path, sift_sfm)

    if not is_valid_colmap_model(sift_sfm):
        raise RuntimeError(
            f"COLMAP model at {sift_sfm} is incomplete. Expected cameras.bin, images.bin, and points3D.bin."
        )


    query_files = list(dataset.glob('queries/*_time_queries_with_intrinsics.txt'))
    if not query_files:
        query_files = list(dataset.glob('*_time_queries_with_intrinsics.txt'))
        
    if not query_files:
        raise FileNotFoundError(f"Missing query text files in {dataset}")

    combined_queries = outputs / f'aachen_{version}_combined_queries.txt'
    if not combined_queries.exists():
        with open(combined_queries, 'w') as f_out:
            for q_file in query_files:
                with open(q_file, 'r') as f_in:
                    f_out.write(f_in.read() + '\n')

    sfm_pairs = outputs / 'pairs-db-covis20.txt'
    loc_pairs = outputs / 'pairs-query-netvlad50.txt'

    if not sfm_pairs.exists():
        print(f"\n--- Generating Database Pairs (Covisibility = 20) ---")
        pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=20)

    if not loc_pairs.exists():
        print(f"\n--- Generating Query Pairs (NetVLAD Retrieval = 50) ---")
        retrieval_conf = extract_features.confs['netvlad']
        retrieval_path = extract_features.main(retrieval_conf, images, outputs)
        pairs_from_retrieval.main(
            retrieval_path, loc_pairs, num_matched=50, 
            query_list=combined_queries, db_model=sift_sfm
        )


    xfeat_conf = {
        'output': f'feats',
        'model': {
            'name': 'custom_xfeat', 
            'max_keypoints': args.top_k,
            'weights': args.checkpoint,
            'use_original': args.use_original,
            'exp': args.exp
        },
        'preprocessing': {'grayscale': False, 'resize_max': 1024}
    }
    match_conf = {
        'output': f'matches',
        'model': {'name': 'nearest_neighbor', 'mutual_check': True}
    }

    print("\n[1/5] Extracting Features...")
    feature_path = extract_features.main(xfeat_conf, images, outputs)
    
    print("\n[2/5] Matching Database Pairs...")
    match_path = match_features.main(match_conf, sfm_pairs, xfeat_conf['output'], outputs)
    
    print("\n[3/5] Triangulating Reference SfM Model...")
    reference_sfm = outputs / f'sfm'
    triangulation.main(reference_sfm, sift_sfm, images, sfm_pairs, feature_path, match_path)
    
    print("\n[4/5] Matching Query Pairs...")
    match_features.main(match_conf, loc_pairs, xfeat_conf['output'], outputs, match_path)
    
    print("\n[5/5] Localizing Queries...")
    results_txt = outputs / f'Aachen_{version}_hloc.txt'
    localize_sfm.main(reference_sfm, combined_queries, loc_pairs, feature_path, match_path, results_txt)

    print("\n" + "="*80)
    print(" Localization pipeline complete!")
    print(f" Poses saved to: {results_txt}")
    print("-" * 80)
    print(f" To evaluate, upload '{results_txt.name}' to:")
    print(f" https://www.visuallocalization.net/submission/ (Track: Aachen Day-Night {version})")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()