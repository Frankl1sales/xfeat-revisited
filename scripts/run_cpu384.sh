#!/usr/bin/env bash
BASE="/media/frank/01DBA5D4E1C670D06/Feature_Matching"
REPO="$BASE/xfeat-revisited"
BR="$BASE/luckfox-xfeat-quantization/_docker_bridge_scannet_cpu"
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_scannet_cpu
mkdir -p "$BR/datasets" "$BR/output"
cp -r "$REPO/modules" "$REPO/assets" "$REPO/checkpoints" "$REPO/third_party" "$BR/"
cp -al "$REPO/datasets/ScanNet1500" "$BR/datasets/"
docker exec -w /workspace/_docker_bridge_scannet_cpu xfeat_exportador \
  /usr/bin/python3 -m modules.evaluation.eval \
    --dataset scannet1500 --matcher xfeat --use-original \
    --checkpoint checkpoints/xfeat.pt --data-root datasets/ScanNet1500 \
    --ransac-thr 0.5 --resize-hw 288 384 \
    --output-dir output/scannet_fp32_384x288 --num-workers 0
docker exec xfeat_exportador chmod -R a+rwX /workspace/_docker_bridge_scannet_cpu/output
cp -r "$BR/output/"* "$REPO/output/"
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_scannet_cpu
