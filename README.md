

# XFeat Reproduction and Ablations

This repository contains a reproduction and extension of the CVPR 2024 paper: **[XFeat: Accelerated Features for Lightweight Image Matching](https://arxiv.org/abs/2404.19174)**. 

In addition to the original XFeat experiments, this codebase includes re-implemented visualization and evaluations scripts (including Aachen Day-Night and HPatches), and a comprehensive set of architectural ablation studies.

## Table of Contents
- [XFeat Reproduction and Ablations](#xfeat-reproduction-and-ablations)
  - [Table of Contents](#table-of-contents)
  - [Setup and Installation](#setup-and-installation)
  - [Datasets](#datasets)
  - [Training \& Ablation Studies](#training--ablation-studies)
    - [Ablation Architectures](#ablation-architectures)
    - [Training Command](#training-command)
  - [Evaluation](#evaluation)
    - [MegaDepth-1500 \& ScanNet-1500](#megadepth-1500--scannet-1500)
    - [Aachen Day-Night](#aachen-day-night)
    - [HPatches](#hpatches)
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

# Update submodules
git submodule update --init --recursive

# Create and activate conda env
conda create -n xfeat python=3.8
conda activate xfeat

# Install PyTorch (Modify for your specific CUDA version if using GPU)
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Install required dependencies for evaluation and plotting
pip install opencv-contrib-python-headless==4.10.0.84 poselib kornia==0.7.2 tqdm pandas matplotlib numpy
```

## Datasets

Training and evaluation datasets can be downloaded to `datasets` using the following commands.

**Note:** These commands populate the default paths expected by the repo scripts (for example `datasets/coco_20k`, `datasets/MegaDepth`, `datasets/Mega1500`, `datasets/ScanNet1500`, `datasets/hpatches-sequences-release`, and `datasets/aachen`).

Run each command separately.
```bash
# Training datasets
python3 -m modules.datasets.download --coco
python3 -m modules.datasets.download --megadepth

# Evaluation datasets
python3 -m modules.datasets.download --megadepth1500
python3 -m modules.datasets.download --scannet1500
python3 -m modules.datasets.download --hpatches
python3 -m modules.datasets.download --aachen
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
You can train the base model or any of the ablations using the following command:

**Note:** `ALIKE` is required by the training losses. Run `git submodule update --init --recursive` after cloning so `third_party/ALIKE` is available before training.

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

We provide evaluation scripts for multiple standard matching and localization benchmarks. Add `--use-original` to any of the evaluation commands below to run the original XFeat implementation.

### MegaDepth-1500 & ScanNet-1500

To evaluate a trained checkpoint on both `xfeat` and `xfeat-star` modes:

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

Aachen Day-Night evaluates visual localization under challenging conditions (day-to-night illumination changes).

**Note:** `hloc` is required for Aachen evaluation. Run `git submodule update --init --recursive` after cloning so `third_party/hloc` is available before running `eval_aachen`.

```bash
python3 -m modules.evaluation.eval_aachen \
  --data-root path/to/aachen \
  --checkpoint path/to/checkpoint \
  --top-k 4096 \
  --exp skip_default
```

### HPatches

HPatches evaluates descriptor matching under structured viewpoint and illumination variations.

```bash
python3 -m modules.evaluation.eval_hpatches \
  --data-root path/to/hpatches-sequences-release \
  --checkpoint path/to/checkpoint \
  --matcher xfeat \
  --ransac-thr 1.5 \
  --top-k 4096 \
  --exp skip_default
```

## Running Visualizations & Demos

We provide lightweight entrypoints to reproduce matching results and benchmark performance.

### Hardware Benchmark Plots
Generate a comparative Pareto curve plot comparing frames per second (FPS) and accuracy across different hardware setups (e.g., Mac vs. Windows).

```bash
# Run the timing benchmark module to produce CSV results
python -m timing.time

# Plot combined benchmark CSVs
python -m timing.plot --mac_file benchmark_results_MAC.csv --win_file benchmark_results_WIN.csv --output benchmark_plot_combined.png
```
*(Use `--smoke-test` on timing benchmark module to check dependencies before running full benchmark).*

### Image Matching & Warping
Run XFeat and XFeat* on an image pair to visualize semi-dense feature matching and homography-based planar warping.

```bash
python -m match --img1 ./assets/frame1.png --img2 ./assets/frame2.png --output_xfeat semi_dense_matching_xfeat.png --output_xfeat_star semi_dense_matching_xfeat_star.png
```

## Acknowledgements

A huge thank you to the original authors of XFeat: Guilherme Potje, Felipe Cadar, Andre Araujo, Renato Martins, and Erickson R. Nascimento. If you use this code, please be sure to check out their [original repository](https://github.com/verlab/accelerated_features) and cite their CVPR 2024 paper.