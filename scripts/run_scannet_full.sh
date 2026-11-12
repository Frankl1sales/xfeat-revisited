#!/usr/bin/env bash
set -u

BASE="/media/frank/01DBA5D4E1C670D06/Feature_Matching"
REPO="$BASE/xfeat-revisited"
DL_DIR="$BASE/luckfox-xfeat-quantization/_scannet_dl"
BRIDGE="$BASE/luckfox-xfeat-quantization/_docker_bridge_scannet"
LOG="$REPO/output/scannet_full.log"

exec > >(tee -a "$LOG") 2>&1
echo "=========================================================="
echo " INICIANDO SCANNET-1500 COMPLETO: $(date)"
echo "=========================================================="
cd "$REPO"

# 1. Baixar e extrair ScanNet-1500
mkdir -p "$REPO/datasets/ScanNet1500"
if [ ! -f "$REPO/datasets/ScanNet1500/test.npz" ]; then
    wget -c "https://github.com/zju3dv/LoFTR/raw/refs/heads/master/assets/scannet_test_1500/test.npz" \
         -O "$REPO/datasets/ScanNet1500/test.npz"
fi

if ! find "$REPO/datasets/ScanNet1500" -maxdepth 3 -type d -name "scene*" | grep -q .; then
    echo "[$(date +%H:%M:%S)] Baixando test_images.tar do ScanNet-1500..."
    docker exec xfeat_exportador rm -rf /workspace/_scannet_dl
    mkdir -p "$DL_DIR"
    docker exec xfeat_exportador /usr/bin/python3 -m gdown \
        "https://drive.google.com/uc?id=1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd" \
        -O /workspace/_scannet_dl/test_images.tar
    docker exec xfeat_exportador bash -lc "
        tar -xf /workspace/_scannet_dl/test_images.tar -C /workspace/_scannet_dl &&
        rm -f /workspace/_scannet_dl/test_images.tar &&
        chmod -R a+rwX /workspace/_scannet_dl
    "
    cp -rn "$DL_DIR/"* "$REPO/datasets/ScanNet1500/"
    docker exec xfeat_exportador rm -rf /workspace/_scannet_dl
fi

# Garantir compatibilidade de estrutura (com e sem subpasta scannet_test_1500)
if [ -d "$REPO/datasets/ScanNet1500/scannet_test_1500" ]; then
    for s in "$REPO/datasets/ScanNet1500/scannet_test_1500"/scene*; do
        [ -d "$s" ] && ln -sfn "$s" "$REPO/datasets/ScanNet1500/$(basename "$s")"
    done
elif ls "$REPO/datasets/ScanNet1500"/scene* >/dev/null 2>&1; then
    mkdir -p "$REPO/datasets/ScanNet1500/scannet_test_1500"
    for s in "$REPO/datasets/ScanNet1500"/scene*; do
        [ -d "$s" ] && ln -sfn "$s" "$REPO/datasets/ScanNet1500/scannet_test_1500/$(basename "$s")"
    done
fi

# 2. Preparar ponte Docker e rodar FP32 Original + FP32 384x288
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_scannet
mkdir -p "$BRIDGE/datasets" "$BRIDGE/output"
cp -r "$REPO/modules" "$REPO/assets" "$REPO/checkpoints" "$REPO/third_party" "$REPO/scripts" "$BRIDGE/"
cp -al "$REPO/datasets/ScanNet1500" "$BRIDGE/datasets/"

echo "[$(date +%H:%M:%S)] Rodando ScanNet-1500 FP32 Original (top_k=4096)..."
docker exec -w /workspace/_docker_bridge_scannet xfeat_exportador \
  /usr/bin/python3 -m modules.evaluation.eval \
    --dataset scannet1500 --matcher xfeat --use-original \
    --checkpoint checkpoints/xfeat.pt --data-root datasets/ScanNet1500 \
    --ransac-thr 0.5 --output-dir output/scannet_fp32_orig --num-workers 0

echo "[$(date +%H:%M:%S)] Rodando ScanNet-1500 FP32 384x288 (Baseline A)..."
docker exec -w /workspace/_docker_bridge_scannet xfeat_exportador \
  /usr/bin/python3 -m modules.evaluation.eval \
    --dataset scannet1500 --matcher xfeat --use-original \
    --checkpoint checkpoints/xfeat.pt --data-root datasets/ScanNet1500 \
    --ransac-thr 0.5 --resize-hw 288 384 --output-dir output/scannet_fp32_384x288 --num-workers 0

docker exec xfeat_exportador chmod -R a+rwX /workspace/_docker_bridge_scannet/output
cp -r "$BRIDGE/output/"* "$REPO/output/"
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_scannet

# 3. Extrair features na placa Luckfox RV1103
echo "[$(date +%H:%M:%S)] Extraindo ScanNet-1500 na placa RV1103..."
python3 "$REPO/scripts/extract_board_all.py" --dataset scannet

# 4. Avaliar placa e atualizar relatório consolidado (com MegaDepth, HPatches e ScanNet)
echo "[$(date +%H:%M:%S)] Avaliando CSVs do ScanNet e gerando relatório final..."
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_scannet
mkdir -p "$BRIDGE/datasets" "$BRIDGE/output"
cp -r "$REPO/modules" "$REPO/assets" "$REPO/scripts" "$BRIDGE/"
cp -al "$REPO/datasets/hpatches-sequences-release" "$BRIDGE/datasets/" 2>/dev/null || true
cp -al "$REPO/datasets/ScanNet1500" "$BRIDGE/datasets/" 2>/dev/null || true
cp -al "$REPO/datasets/board_features" "$BRIDGE/datasets/" 2>/dev/null || true
cp -r "$REPO/output/"* "$BRIDGE/output/"

docker exec -w /workspace/_docker_bridge_scannet xfeat_exportador \
  /usr/bin/python3 scripts/eval_board_all.py

docker exec xfeat_exportador chmod -R a+rwX /workspace/_docker_bridge_scannet/output
cp -r "$BRIDGE/output/"* "$REPO/output/"
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_scannet

echo "=========================================================="
echo " SCANNET-1500 CONCLUÍDO EM: $(date)"
echo "=========================================================="
