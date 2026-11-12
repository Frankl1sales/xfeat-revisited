#!/usr/bin/env bash
set -u

BASE="/media/frank/01DBA5D4E1C670D06/Feature_Matching"
REPO="$BASE/xfeat-revisited"
BRIDGE="$BASE/luckfox-xfeat-quantization/_docker_bridge_fp32"
LOG="$REPO/output/fp32_remaining.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== INICIANDO COMPLEMENTO FP32: $(date) ==="
cd "$REPO"

# Passo A: Baixar ScanNet-1500 primeiro (para a placa encontrar pronto)
mkdir -p "$REPO/datasets/ScanNet1500"
if [ ! -f "$REPO/datasets/ScanNet1500/test.npz" ]; then
    wget -c "https://github.com/zju3dv/LoFTR/raw/refs/heads/master/assets/scannet_test_1500/test.npz" \
         -O "$REPO/datasets/ScanNet1500/test.npz"
fi

if [ ! -d "$REPO/datasets/ScanNet1500/scannet_test_1500" ]; then
    docker exec xfeat_exportador /usr/bin/python3 -m pip install -q -U gdown || true
    TAR_PATH="$REPO/datasets/ScanNet1500/test_images.tar"
    docker exec -v "$REPO/datasets/ScanNet1500:/scannet_dl" xfeat_exportador \
        /usr/bin/python3 -m gdown "https://drive.google.com/uc?id=1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd" -O /scannet_dl/test_images.tar 2>/dev/null || \
    docker exec xfeat_exportador bash -lc "
        mkdir -p /workspace/_scannet_tmp &&
        /usr/bin/python3 -m gdown 'https://drive.google.com/uc?id=1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd' -O /workspace/_scannet_tmp/test_images.tar &&
        tar -xf /workspace/_scannet_tmp/test_images.tar -C /workspace/_scannet_tmp &&
        rm -f /workspace/_scannet_tmp/test_images.tar &&
        chmod -R a+rwX /workspace/_scannet_tmp
    "
    if [ -d "$BASE/luckfox-xfeat-quantization/_scannet_tmp" ]; then
        cp -rn "$BASE/luckfox-xfeat-quantization/_scannet_tmp/"* "$REPO/datasets/ScanNet1500/"
        docker exec xfeat_exportador rm -rf /workspace/_scannet_tmp
    fi
    if [ ! -d "$REPO/datasets/ScanNet1500/scannet_test_1500" ] && ls "$REPO/datasets/ScanNet1500/scene"* >/dev/null 2>&1; then
        ln -sfn "$REPO/datasets/ScanNet1500" "$REPO/datasets/ScanNet1500/scannet_test_1500"
    fi
fi

# Passo B: Preparar ponte limpa no container
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_fp32
mkdir -p "$BRIDGE/datasets" "$BRIDGE/output"
cp -r "$REPO/modules" "$REPO/assets" "$REPO/checkpoints" "$REPO/third_party" "$REPO/scripts" "$BRIDGE/"
cp -al "$REPO/datasets/hpatches-sequences-release" "$BRIDGE/datasets/"
[ -d "$REPO/datasets/ScanNet1500" ] && cp -al "$REPO/datasets/ScanNet1500" "$BRIDGE/datasets/"

# Passo C: HPatches FP32 Original e 384x288
echo "[$(date +%H:%M:%S)] Rodando HPatches FP32 Original e 384x288..."
docker exec -w /workspace/_docker_bridge_fp32 xfeat_exportador \
  /usr/bin/python3 scripts/eval_hpatches_fp32.py --use-original \
    --output-json output/hpatches_fp32_orig/hpatches_fp32.json
docker exec -w /workspace/_docker_bridge_fp32 xfeat_exportador \
  /usr/bin/python3 scripts/eval_hpatches_fp32.py --use-original \
    --resize-hw 288 384 --output-json output/hpatches_fp32_384x288/hpatches_fp32.json

# Passo D: ScanNet-1500 FP32 Original e 384x288
if [ -f "$REPO/datasets/ScanNet1500/test.npz" ] && [ -d "$REPO/datasets/ScanNet1500/scannet_test_1500" ]; then
    echo "[$(date +%H:%M:%S)] Rodando ScanNet-1500 FP32 Original e 384x288..."
    docker exec -w /workspace/_docker_bridge_fp32 xfeat_exportador \
      /usr/bin/python3 -m modules.evaluation.eval \
        --dataset scannet1500 --matcher xfeat --use-original \
        --checkpoint checkpoints/xfeat.pt --data-root datasets/ScanNet1500 \
        --ransac-thr 0.5 --output-dir output/scannet_fp32_orig --num-workers 0
    docker exec -w /workspace/_docker_bridge_fp32 xfeat_exportador \
      /usr/bin/python3 -m modules.evaluation.eval \
        --dataset scannet1500 --matcher xfeat --use-original \
        --checkpoint checkpoints/xfeat.pt --data-root datasets/ScanNet1500 \
        --ransac-thr 0.5 --resize-hw 288 384 --output-dir output/scannet_fp32_384x288 --num-workers 0
fi

# Passo E: Copiar JSONs gerados de volta e atualizar relatório final
docker exec xfeat_exportador chmod -R a+rwX /workspace/_docker_bridge_fp32/output
cp -r "$BRIDGE/output/"* "$REPO/output/"
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_fp32

# Aguarda a placa terminar caso ainda esteja extraindo e consolida a tabela final
while pgrep -f "extract_board" >/dev/null 2>&1; do
    sleep 20
done
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_rep
mkdir -p "$BRIDGE"_rep/datasets "$BRIDGE"_rep/output
cp -r "$REPO/modules" "$REPO/assets" "$REPO/scripts" "$BRIDGE"_rep/
cp -al "$REPO/datasets/hpatches-sequences-release" "$BRIDGE"_rep/datasets/ 2>/dev/null || true
cp -al "$REPO/datasets/ScanNet1500" "$BRIDGE"_rep/datasets/ 2>/dev/null || true
cp -al "$REPO/datasets/board_features" "$BRIDGE"_rep/datasets/ 2>/dev/null || true
cp -r "$REPO/output/"* "$BRIDGE"_rep/output/
docker exec -w /workspace/_docker_bridge_rep xfeat_exportador /usr/bin/python3 scripts/eval_board_all.py
docker exec xfeat_exportador chmod -R a+rwX /workspace/_docker_bridge_rep/output
cp -r "$BRIDGE"_rep/output/* "$REPO/output/"
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_rep

echo "=== COMPLEMENTO FP32 E RELATÓRIO FINAL CONCLUÍDOS: $(date) ==="
