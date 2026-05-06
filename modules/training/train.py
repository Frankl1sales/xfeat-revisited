"""
	Training code modified from original
"""
import argparse
import os
import time
import sys
import glob
import random

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tqdm

from modules.model import *
from modules.datasets.megadepth import *
from modules.datasets.augmentation import AugmentationPipe
from modules.training.utils import *
from modules.training.losses import *

def parse_args():
    parser = argparse.ArgumentParser(description="XFeat training script.")

    parser.add_argument('--megadepth_root_path', type=str, default='datasets/MegaDepth',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str, default='datasets/coco_20k',
                        help='Path to the synthetic dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, default='.',
                        help='Path to save the checkpoints.')
    parser.add_argument('--training_type', type=str, default='xfeat_default',
                        choices=['xfeat_default', 'xfeat_synthetic', 'xfeat_megadepth'],
                        help='Training scheme. xfeat_default uses both megadepth & synthetic warps.')
    parser.add_argument('--exp', type=str, default='skip_default',
                        help='Experiment name to append to logs and checkpoints.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for training. Default is 100.')
    parser.add_argument('--n_steps', type=int, default=16000,
                        help='Number of training steps. Default is 16000.')
    parser.add_argument('--lr', type=float, default=3e-4*10,
                        help='Learning rate. Default is 0.0003.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(800, 608), help='Training resolution as width,height. Default is (800, 608).')
    parser.add_argument('--device_num', type=str, default='0',
                        help='Device number to use for training. Default is "0".')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, perform a dry run training with a mini-batch for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=1000,
                        help='Save checkpoints every N steps. Default is 500.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    return args


class Trainer():

    def __init__(self, megadepth_root_path, 
                       synthetic_root_path, 
                       ckpt_save_path, 
                       model_name = 'xfeat_default',
                       exp_name = '',
                       batch_size = 10, n_steps = 160_000, lr= 3e-4, gamma_steplr=0.5, 
                       training_res = (800, 608), device_num="0", dry_run = False,
                       save_ckpt_every = 500):

        self.dev = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel(ablation_type=exp_name).to(self.dev)

        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()) , lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=gamma_steplr)

        if model_name in ('xfeat_default', 'xfeat_synthetic'):
            self.augmentor = AugmentationPipe(
                                        img_dir = synthetic_root_path,
                                        device = self.dev, load_dataset = True,
                                        batch_size = int(self.batch_size * 0.4 if model_name=='xfeat_default' else batch_size),
                                        out_resolution = training_res, 
                                        warp_resolution = training_res,
                                        sides_crop = 0.1,
                                        max_num_imgs = 3_000,
                                        num_test_imgs = 5,
                                        photometric = True,
                                        geometric = True,
                                        reload_step = 4_000
                                        )
        else:
            self.augmentor = None

        if model_name in ('xfeat_default', 'xfeat_megadepth'):
            TRAIN_BASE_PATH = f"{megadepth_root_path}/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/phoenix/S6/zl548/MegaDepth_v1/"
            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            data = torch.utils.data.ConcatDataset( [MegaDepthDataset(root_dir = TRAINVAL_DATA_SOURCE,
                            npz_path = path) for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")] )

            self.data_loader = DataLoader(
                data,
                batch_size=int(self.batch_size * 0.6 if model_name=='xfeat_default' else batch_size),
                shuffle=True,
                num_workers=2,  # Reduce to avoid HDF5 file lock issues
                pin_memory=False  # Disable to reduce memory usage
            )
            self.data_iter = iter(self.data_loader)
        else:
            self.data_iter = None

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
        
        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.run_name = f"{model_name}_{exp_name}" if exp_name else model_name
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{self.run_name}_' + time_str)
        self.model_name = model_name
        self.save_name = f"{model_name}_{exp_name}_{time_str}" if exp_name else f"{model_name}_{time_str}"



    def train(self):
        self.net.train()
        difficulty = 0.10
        p1s, p2s, H1, H2 = None, None, None, None
        d = None

        # Loss weights (can be tuned)
        alpha, beta, gamma, delta = 1.0, 1.0, 1.0, 1.0

        if self.augmentor is not None:
            p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)
        if self.data_iter is not None:
            d = next(self.data_iter)

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                if not self.dry_run:
                    if self.data_iter is not None:
                        try:
                            # Get the next MD batch
                            d = next(self.data_iter)
                        except StopIteration:
                            print("End of DATASET!")
                            # If StopIteration is raised, create a new iterator.
                            self.data_iter = iter(self.data_loader)
                            d = next(self.data_iter)
                    if self.augmentor is not None:
                        # Grab synthetic data
                        p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

                if d is not None:
                    for k in d.keys():
                        if isinstance(d[k], torch.Tensor):
                            d[k] = d[k].to(self.dev)
                    p1, p2 = d['image0'], d['image1']
                    positives_md_coarse = spvs_coarse(d, 8)

                if self.augmentor is not None:
                    h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                    _, positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse)

                # Join megadepth & synthetic data
                with torch.inference_mode():
                    # RGB -> GRAY
                    if d is not None:
                        p1 = p1.mean(1, keepdim=True)
                        p2 = p2.mean(1, keepdim=True)
                    if self.augmentor is not None:
                        p1s = p1s.mean(1, keepdim=True)
                        p2s = p2s.mean(1, keepdim=True)

                    # Cat two batches
                    if self.model_name in ('xfeat_default'):
                        p1 = torch.cat([p1s, p1], dim=0)
                        p2 = torch.cat([p2s, p2], dim=0)
                        positives_c = positives_s_coarse + positives_md_coarse
                    elif self.model_name in ('xfeat_synthetic'):
                        p1 = p1s; p2 = p2s
                        positives_c = positives_s_coarse
                    else:
                        positives_c = positives_md_coarse

                # Check if batch is corrupted with too few correspondences
                is_corrupted = False
                for p in positives_c:
                    if len(p) < 30:
                        is_corrupted = True
                if is_corrupted:
                    continue

                # Forward pass
                feats1, kpts1, hmap1 = self.net(p1)
                feats2, kpts2, hmap2 = self.net(p2)

                acc_coarse_0 = None
                total_loss = 0.0
                for b in range(len(positives_c)):
                    # Get positive correspondencies
                    pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                    # Grab features at corresponding idxs
                    m1 = feats1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)
                    m2 = feats2[b, :, pts2[:,1].long(), pts2[:,0].long()].permute(1,0)

                    # Grab reliability maps at corresponding idxs
                    r1 = hmap1[b, 0, pts1[:,1].long(), pts1[:,0].long()]
                    r2 = hmap2[b, 0, pts2[:,1].long(), pts2[:,0].long()]

                    # Dual-softmax loss (Paper Eq. 3)
                    l_ds, conf1, conf2 = dual_softmax_loss(m1, m2)
                    # Reliability loss (Paper Eq. 4)
                    l_rel = reliability_loss(r1, r2, conf1, conf2)

                    # Fine loss (paper Eq. 5)
                    l_fine = fine_loss(
                        feats1[b], feats2[b], pts1, pts2, self.net.fine_matcher, stride=8
                    )

                    # Keypoint loss (average both directions)
                    l_kp1, acc1 = keypoint_loss(kpts1[b], p1[b])
                    l_kp2, acc2 = keypoint_loss(kpts2[b], p2[b])
                    l_kp = (l_kp1 + l_kp2) / 2

                    # Weighted sum
                    loss = alpha * l_ds + beta * l_rel + gamma * l_fine + delta * l_kp

                    total_loss = total_loss + loss

                    # For the first batch, log coarse accuracy
                    if b == 0:
                        acc_coarse_0 = check_accuracy(m1, m2)

                # Backward and optimize (once per batch)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i+1) % self.save_ckpt_every == 0:
                    print('saving iter ', i+1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.save_name}_{i+1}.pth')

                # Log coarse accuracy if available
                if acc_coarse_0 is not None:
                    self.writer.add_scalar('Accuracy/coarse', acc_coarse_0, i)

                pbar.set_description(
                    f"Loss: {loss.item():.4f} ds: {l_ds.item():.4f} rel: {l_rel.item():.4f} fine: {l_fine.item():.4f} kp: {l_kp.item():.4f} acc_c: {acc_coarse_0:.3f}"
                )
                pbar.update(1)

        # Explicitly save the final model state after training
        print('Saving final model state...')
        torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.save_name}_final.pth')


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed set to: {args.seed}")

    trainer = Trainer(
        megadepth_root_path=args.megadepth_root_path, 
        synthetic_root_path=args.synthetic_root_path, 
        ckpt_save_path=args.ckpt_save_path,
        model_name=args.training_type,
        exp_name=args.exp,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every
    )

    trainer.train()
