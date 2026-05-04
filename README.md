

# XFeat Reproduction and Ablations

This repository contains a reproduction and extension of the CVPR 2024 paper: **[XFeat: Accelerated Features for Lightweight Image Matching](https://arxiv.org/abs/2404.19174)**. 

In addition to the original XFeat experiments, this codebase includes re-implemented visualization and evaluations scripts (including Aachen Day-Night and HPatches), and a comprehensive set of architectural ablation studies.

## Table of Contents
- [XFeat Reproduction and Ablations](#xfeat-reproduction-and-ablations)
  - [Table of Contents](#table-of-contents)
  - [Setup and Installation](#setup-and-installation)
  - [Training \& Ablation Studies](#training--ablation-studies)
    - [Ablation Architectures](#ablation-architectures)
    - [Training Command](#training-command)
  - [Evaluation](#evaluation)
    - [MegaDepth-1500 \& ScanNet-1500](#megadepth-1500--scannet-1500)
    - [Aachen Day-Night](#aachen-day-night)
    - [HPatches Benchmark](#hpatches-benchmark)
  - [Running Visualizations \& Demos](#running-visualizations--demos)
    - [Hardware Benchmark Plots](#hardware-benchmark-plots)
    - [Image Matching \& Warping](#image-matching--warping)
  - [Acknowledgements](#acknowledgements)

## Setup and Installation

XFeat has minimal dependencies. We recommend using a Conda environment.

```bash
# Clone this repository
git clone https://github.com/GalaxyGHz/xfeat-reproduction.git
cd xfeat-reproduction

# Create and activate conda env
conda create -n xfeat python=3.8
conda activate xfeat

# Install PyTorch (Modify for your specific CUDA version if using GPU)
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Install required dependencies for evaluation and plotting
pip install opencv-contrib-python-headless==4.10.0.84 poselib kornia==0.7.2 tqdm pandas matplotlib numpy
```

## Training & Ablation Studies

### Ablation Architectures
To thoroughly evaluate the design choices of XFeat, we introduce the `--exp` flag during training, which controls the underlying architectural variant. The supported experiments are:

- `skip_default`: Original architecture (1 skip connection to block 4).
- `skip_none`: **Ablation 1** - Removes the single skip-connection entirely.
- `skip_input`: **Ablation 2** - Adds skip-connections from the input directly to EACH block.
- `skip_resnet`: **Ablation 3** - Adds ResNet-like residual skip-connections to each block.
- `non_parallel_keypoints`: **Ablation 4** - Replaces the parallel keypoint detector with a sequential block that operates directly on the encoder feature map.

### Training Command
Ensure your datasets (MegaDepth, COCO) are downloaded and formatted correctly. You can train the base model or any of the ablations using the following command:

```bash
python3 -m modules.training.train \
  --batch_size 10 \
  --n_steps 160000 \
  --exp skip_default \
  --ckpt_save_path ./ablations/default \
  --save_ckpt_every 10000
```
*(Change `--exp skip_default` to your desired ablation type).*

## Evaluation

We provide evaluation scripts for multiple standard matching and localization benchmarks. 

### MegaDepth-1500 & ScanNet-1500

First, download the standard 1500 evaluation pairs for both datasets. To evaluate a trained checkpoint on both `xfeat` and `xfeat-star` modes:

**MegaDepth-1500:**
```bash
# Evaluate standard XFeat
python3 -m modules.evaluation.eval \
  --dataset megadepth1500 \
  --matcher xfeat \
  --checkpoint path/to/checkpoint \
  --ransac-thr 2.5 \
  --output-dir output/path \
  --exp skip_default \
  --num-workers 8 \
  --cache-images

# Evaluate XFeat-Star
python3 -m modules.evaluation.eval \
  --dataset megadepth1500 \
  --matcher xfeat-star \
  --checkpoint path/to/checkpoint \
  --ransac-thr 2.5 \
  --output-dir output/path \
  --exp skip_default \
  --num-workers 8 \
  --cache-images
```

*(Swap `--dataset megadepth1500` with `--dataset scannet1500` to evaluate on ScanNet).*

### Aachen Day-Night

The Aachen Day-Night dataset evaluates visual localization under challenging conditions (day-to-night illumination changes).

**Dataset Preparation:**
You can set up the Aachen Day-Night dataset (version 1.0) utilizing the structure defined by [Hierarchical-Localization (hloc)](https://github.com/cvg/Hierarchical-Localization). Download the images and reference poses into your desired `--data-root` directory.

**Evaluation Command:**
```bash
CUDA_VISIBLE_DEVICES=1 python -m modules.evaluation.eval_aachen \
  --data-root path/to/aachen \
  --checkpoint path/to/checkpoint \
  --top-k 4096 \
  --exp skip_default \
  --use-original # (optional to evaluate original model implementation)
```

### HPatches Benchmark

HPatches evaluates descriptor matching under structured viewpoint and illumination variations.

**Dataset Preparation:**
Download the `hpatches-sequences-release` dataset.

**Evaluation Command:**
```bash
CUDA_VISIBLE_DEVICES=2 python -m modules.evaluation.eval_hpatches \
  --data-root path/to/hpatches-sequences-release \
  --checkpoint path/to/checkpoint \
  --matcher xfeat-star \
  --ransac-thr 1.5 \
  --top-k 4096 \
  --exp skip_default
  --use-original # (optional to evaluate original model implementation)
```

## Running Visualizations & Demos

We provide standalone Python scripts to easily reproduce original matching results and hardware benchmark performances.

### Hardware Benchmark Plots
Generate a comparative Pareto curve plot comparing frames per second (FPS) and accuracy across different hardware setups (e.g., Mac vs. Windows).

```bash
python plot_benchmarks.py \
  --mac_file benchmark_results_MAC.csv \
  --win_file benchmark_results_WIN.csv \
  --output benchmark_plot_combined.png
```

### Image Matching & Warping
Run XFeat and XFeat* on an image pair to visualize semi-dense feature matching and homography-based planar warping.

```bash
python match_images.py \
  --img1 ./assets/frame1.png \
  --img2 ./assets/frame2.png \
  --output_xfeat semi_dense_matching_xfeat.png \
  --output_xfeat_star semi_dense_matching_xfeat_star.png
```

## Acknowledgements

A huge thank you to the original authors of XFeat: Guilherme Potje, Felipe Cadar, Andre Araujo, Renato Martins, and Erickson R. Nascimento. If you use this code, please be sure to check out their [original repository](https://github.com/verlab/accelerated_features) and cite their CVPR 2024 paper.