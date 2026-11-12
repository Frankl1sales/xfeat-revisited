# Benchmark XFeat: Luckfox RV1103 (INT8 Split) vs PyTorch FP32

## 1. MegaDepth-1500 (Outdoor Relative Pose, RANSAC 2.5px)

| Configuração | AUC@5 | AUC@10 | AUC@20 | Acc@10° | MIR | Mean Inliers |
|---|---|---|---|---|---|---|
| PyTorch FP32 Original (top_k=4096) | 43.24% | 57.12% | 68.22% | 75.33% | 56.28% | 915.9 |
| PyTorch FP32 384x288 (Baseline A) | 12.22% | 22.34% | 35.80% | 39.47% | 63.28% | 323.2 |
| Luckfox RV1103 INT8 384x288 (allkpts) | 3.90% | 8.97% | 18.49% | 18.73% | 42.49% | 67.8 |

## 2. HPatches (Homography Estimation, MAGSAC 1.5px)

| Configuração | Illum MHA@3 | Illum MHA@5 | Illum MHA@7 | View MHA@3 | View MHA@5 | View MHA@7 | Mean Inliers |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 Original (top_k=4096) | 94.74% | 97.19% | 97.89% | 67.80% | 81.36% | 86.10% | 870.9 |
| PyTorch FP32 384x288 (Baseline A) | 93.33% | 96.49% | 96.49% | 50.85% | 67.80% | 74.92% | 275.7 |
| Luckfox RV1103 INT8 384x288 (allkpts) | 89.12% | 94.74% | 95.09% | 17.63% | 36.27% | 47.12% | 47.7 |

## 3. ScanNet-1500 (Indoor Relative Pose, RANSAC 0.5px)

| Configuração | AUC@5 | AUC@10 | AUC@20 |
|---|---|---|---|
| PyTorch FP32 Original (top_k=4096) | 10.89% | 23.58% | 38.50% |
| PyTorch FP32 384x288 (Baseline A) | 10.89% | 23.58% | 38.50% |
| Luckfox RV1103 INT8 384x288 (allkpts) | 0.00% | 0.00% | 0.00% |
