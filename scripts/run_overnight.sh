#!/usr/bin/env bash
set -u

BASE="/media/frank/01DBA5D4E1C670D06/Feature_Matching"
REPO="$BASE/xfeat-revisited"
BRIDGE="$BASE/luckfox-xfeat-quantization/_docker_bridge"
LOG="$REPO/output/overnight.log"

exec > >(tee -a "$LOG") 2>&1
echo "=========================================================="
echo " INICIANDO BENCHMARK AUTOMÁTICO: $(date)"
echo "=========================================================="

cd "$REPO"

sync_to_bridge() {
    docker exec xfeat_exportador rm -rf /workspace/_docker_bridge
    mkdir -p "$BRIDGE/datasets" "$BRIDGE/output"
    cp -r "$REPO/modules" "$REPO/assets" "$REPO/checkpoints" "$REPO/third_party" "$REPO/scripts" "$BRIDGE/"
    [ -d "$REPO/datasets/Mega1500" ] && cp -al "$REPO/datasets/Mega1500" "$BRIDGE/datasets/" 2>/dev/null || true
    [ -d "$REPO/datasets/hpatches-sequences-release" ] && cp -al "$REPO/datasets/hpatches-sequences-release" "$BRIDGE/datasets/" 2>/dev/null || true
    [ -d "$REPO/datasets/ScanNet1500" ] && cp -al "$REPO/datasets/ScanNet1500" "$BRIDGE/datasets/" 2>/dev/null || true
    [ -d "$REPO/datasets/board_features" ] && cp -al "$REPO/datasets/board_features" "$BRIDGE/datasets/" 2>/dev/null || true
    [ -d "$REPO/output" ] && cp -r "$REPO/output/"* "$BRIDGE/output/" 2>/dev/null || true
}

sync_from_bridge() {
    docker exec xfeat_exportador chmod -R a+rwX /workspace/_docker_bridge/output /workspace/_docker_bridge/datasets 2>/dev/null || true
    cp -r "$BRIDGE/output/"* "$REPO/output/" 2>/dev/null || true
}

# --- ETAPA 1: MegaDepth-1500 FP32 Original ---
echo "[$(date +%H:%M:%S)] ETAPA 1: MegaDepth-1500 FP32 Original..."
sync_to_bridge
docker exec -w /workspace/_docker_bridge xfeat_exportador \
  /usr/bin/python3 -m modules.evaluation.eval \
    --dataset megadepth1500 --matcher xfeat --use-original \
    --checkpoint checkpoints/xfeat.pt --data-root datasets/Mega1500 \
    --ransac-thr 2.5 --output-dir output/megadepth_fp32_orig --num-workers 0 || true
sync_from_bridge

# --- ETAPA 2: Preparar e avaliar HPatches FP32 (Original e 384x288) ---
echo "[$(date +%H:%M:%S)] ETAPA 2: Preparando HPatches..."
OLD_HP="/media/frank/01DBA5D4E1C670D05/RV1103_SUCESS/xfeat-rv1103/evaluation/scripts/dataset/hpatches-sequences-release"
if [ ! -d "$REPO/datasets/hpatches-sequences-release/i_ajuntament" ]; then
    if [ -d "$OLD_HP/i_ajuntament" ]; then
        ln -sfn "$OLD_HP" "$REPO/datasets/hpatches-sequences-release"
    else
        sync_to_bridge
        docker exec -w /workspace/_docker_bridge xfeat_exportador \
          /usr/bin/python3 -m modules.datasets.download --hpatches --download_dir datasets || true
        docker exec xfeat_exportador chmod -R a+rwX /workspace/_docker_bridge/datasets 2>/dev/null || true
        mv "$BRIDGE/datasets/hpatches-sequences-release" "$REPO/datasets/" 2>/dev/null || true
    fi
fi

if [ -d "$REPO/datasets/hpatches-sequences-release/i_ajuntament" ]; then
    echo "[$(date +%H:%M:%S)] Rodando HPatches FP32 Original e 384x288..."
    sync_to_bridge
    docker exec -w /workspace/_docker_bridge xfeat_exportador \
      /usr/bin/python3 scripts/eval_hpatches_fp32.py --use-original \
        --output-json output/hpatches_fp32_orig/hpatches_fp32.json || true
    docker exec -w /workspace/_docker_bridge xfeat_exportador \
      /usr/bin/python3 scripts/eval_hpatches_fp32.py --use-original \
        --resize-hw 288 384 --output-json output/hpatches_fp32_384x288/hpatches_fp32.json || true
    sync_from_bridge
fi

# --- ETAPA 3: Baixar e avaliar ScanNet-1500 FP32 (Original e 384x288) ---
echo "[$(date +%H:%M:%S)] ETAPA 3: Preparando ScanNet-1500..."
if [ ! -f "$REPO/datasets/ScanNet1500/test.npz" ] || [ ! -d "$REPO/datasets/ScanNet1500/scannet_test_1500" ]; then
    sync_to_bridge
    docker exec -w /workspace/_docker_bridge xfeat_exportador \
      /usr/bin/python3 -m modules.datasets.download --scannet1500 --download_dir datasets || true
    docker exec xfeat_exportador chmod -R a+rwX /workspace/_docker_bridge/datasets 2>/dev/null || true
    mkdir -p "$REPO/datasets/ScanNet1500"
    cp -rn "$BRIDGE/datasets/ScanNet1500/"* "$REPO/datasets/ScanNet1500/" 2>/dev/null || true
    if [ ! -d "$REPO/datasets/ScanNet1500/scannet_test_1500" ] && ls "$REPO/datasets/ScanNet1500/scene"* >/dev/null 2>&1; then
        ln -sfn "$REPO/datasets/ScanNet1500" "$REPO/datasets/ScanNet1500/scannet_test_1500"
    fi
fi

if [ -f "$REPO/datasets/ScanNet1500/test.npz" ] && [ -d "$REPO/datasets/ScanNet1500/scannet_test_1500" ]; then
    echo "[$(date +%H:%M:%S)] Rodando ScanNet-1500 FP32 Original e 384x288..."
    sync_to_bridge
    docker exec -w /workspace/_docker_bridge xfeat_exportador \
      /usr/bin/python3 -m modules.evaluation.eval \
        --dataset scannet1500 --matcher xfeat --use-original \
        --checkpoint checkpoints/xfeat.pt --data-root datasets/ScanNet1500 \
        --ransac-thr 0.5 --output-dir output/scannet_fp32_orig --num-workers 0 || true
    docker exec -w /workspace/_docker_bridge xfeat_exportador \
      /usr/bin/python3 -m modules.evaluation.eval \
        --dataset scannet1500 --matcher xfeat --use-original \
        --checkpoint checkpoints/xfeat.pt --data-root datasets/ScanNet1500 \
        --ransac-thr 0.5 --resize-hw 288 384 --output-dir output/scannet_fp32_384x288 --num-workers 0 || true
    sync_from_bridge
fi

# --- ETAPA 4: Aguardar extração atual da placa (Aba 1) terminar ---
echo "[$(date +%H:%M:%S)] ETAPA 4: Aguardando conclusão da extração MegaDepth na Aba 1..."
while pgrep -f "extract_board_features.py" >/dev/null 2>&1; do
    sleep 20
done
echo "[$(date +%H:%M:%S)] Aba 1 finalizada! Verificando integridade dos CSVs do MegaDepth..."
python3 "$REPO/scripts/extract_board_all.py" --dataset megadepth || true

# --- ETAPA 5: Extrair features da placa para HPatches e ScanNet-1500 ---
if [ -d "$REPO/datasets/hpatches-sequences-release/i_ajuntament" ]; then
    echo "[$(date +%H:%M:%S)] ETAPA 5A: Extraindo HPatches na placa RV1103..."
    python3 "$REPO/scripts/extract_board_all.py" --dataset hpatches || true
fi

if [ -f "$REPO/datasets/ScanNet1500/test.npz" ] && [ -d "$REPO/datasets/ScanNet1500/scannet_test_1500" ]; then
    echo "[$(date +%H:%M:%S)] ETAPA 5B: Extraindo ScanNet-1500 na placa RV1103..."
    python3 "$REPO/scripts/extract_board_all.py" --dataset scannet || true
fi

# --- ETAPA 6: Avaliar todos os CSVs da placa e gerar relatório final ---
echo "[$(date +%H:%M:%S)] ETAPA 6: Calculando métricas finais da placa e gerando relatório..."
sync_to_bridge
docker exec -w /workspace/_docker_bridge xfeat_exportador \
  /usr/bin/python3 scripts/eval_board_all.py || true
sync_from_bridge
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge

echo "=========================================================="
echo " TUDO CONCLUÍDO EM: $(date)"
echo " Relatório salvo em: $REPO/output/BENCHMARK_SUMMARY_REPORT.md"
echo "=========================================================="
