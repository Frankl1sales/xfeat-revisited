#!/usr/bin/env bash
set -u

BASE="/media/frank/01DBA5D4E1C670D06/Feature_Matching"
REPO="$BASE/xfeat-revisited"
BRIDGE="$BASE/luckfox-xfeat-quantization/_docker_bridge_aachen"
LOG="$REPO/output/aachen_full.log"

exec > >(tee -a "$LOG") 2>&1
echo "=========================================================="
echo " FILA AACHEN DAY-NIGHT INICIADA: $(date)"
echo "=========================================================="
cd "$REPO"

# Aguardar conclusão do download do Aachen
while tmux has-session -t aachen_dl 2>/dev/null; do
    sleep 30
done
echo "[$(date +%H:%M:%S)] Download do Aachen concluído!"

# Garantir extração do zip caso ainda não tenha sido extraído
if [ -f "datasets/aachen/images/database_and_query_images.zip" ] && [ ! -d "datasets/aachen/images/images_upright" ]; then
    unzip -qn "datasets/aachen/images/database_and_query_images.zip" -d "datasets/aachen/images"
fi

# Aguardar conclusão do ScanNet (para não disputar CPU/placa)
while tmux has-session -t scannet_bench 2>/dev/null; do
    sleep 30
done

# 1. Preparar ponte Docker e rodar Aachen FP32 Original + FP32 384x288
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_aachen
mkdir -p "$BRIDGE/datasets" "$BRIDGE/output"
cp -r "$REPO/modules" "$REPO/assets" "$REPO/checkpoints" "$REPO/third_party" "$REPO/scripts" "$BRIDGE/"
cp -al "$REPO/datasets/aachen" "$BRIDGE/datasets/"

echo "[$(date +%H:%M:%S)] Rodando Aachen FP32 Original (top_k=4096)..."
docker exec -w /workspace/_docker_bridge_aachen xfeat_exportador \
  /usr/bin/python3 -m modules.evaluation.eval_aachen \
    --data-root datasets/aachen --checkpoint checkpoints/xfeat.pt \
    --use-original --outputs-dir output/aachen_fp32_orig

echo "[$(date +%H:%M:%S)] Rodando Aachen FP32 384x288 (Baseline A)..."
docker exec -w /workspace/_docker_bridge_aachen xfeat_exportador \
  /usr/bin/python3 scripts/eval_aachen_variants.py \
    --mode fp32_384x288 --data-root datasets/aachen \
    --ref-outputs output/aachen_fp32_orig --outputs-dir output/aachen_fp32_384x288

docker exec xfeat_exportador chmod -R a+rwX /workspace/_docker_bridge_aachen/output
cp -r "$BRIDGE/output/"* "$REPO/output/"
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_aachen

# 2. Extrair todas as imagens do Aachen na placa Luckfox RV1103
echo "[$(date +%H:%M:%S)] Extraindo Aachen Day-Night na placa RV1103..."
python3 - << 'PYEOF'
import os, subprocess
from pathlib import Path

repo = Path("/media/frank/01DBA5D4E1C670D06/Feature_Matching/xfeat-revisited")
img_root = repo / "datasets/aachen/images/images_upright"
out_root = repo / "datasets/board_features/aachen"
extractor = Path("/media/frank/01DBA5D4E1C670D06/Feature_Matching/luckfox-xfeat-quantization/extract_board_allkpts.py")

imgs = sorted([p for p in img_root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
print(f"[AACHEN BOARD] Total de imagens: {len(imgs)}")
for idx, p in enumerate(imgs, 1):
    rel = p.relative_to(img_root).as_posix()
    out_csv = out_root / (rel + ".csv")
    if out_csv.exists() and out_csv.stat().st_size > 10:
        continue
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["python3", str(extractor), str(p), str(out_csv)], check=False)
    if idx % 100 == 0 or idx == len(imgs):
        print(f"[AACHEN BOARD {idx}/{len(imgs)}] {rel}")
PYEOF

# 3. Rodar localização hloc com os CSVs da placa RV1103
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_aachen
mkdir -p "$BRIDGE/datasets/board_features" "$BRIDGE/output"
cp -r "$REPO/modules" "$REPO/assets" "$REPO/scripts" "$BRIDGE/"
cp -al "$REPO/datasets/aachen" "$BRIDGE/datasets/"
cp -al "$REPO/datasets/board_features/aachen" "$BRIDGE/datasets/board_features/"
cp -r "$REPO/output/aachen_fp32_orig" "$BRIDGE/output/"

echo "[$(date +%H:%M:%S)] Rodando localização hloc para Luckfox RV1103 INT8..."
docker exec -w /workspace/_docker_bridge_aachen xfeat_exportador \
  /usr/bin/python3 scripts/eval_aachen_variants.py \
    --mode board_int8 --data-root datasets/aachen \
    --ref-outputs output/aachen_fp32_orig --outputs-dir output/aachen_rv1103_int8 \
    --board-csv-root datasets/board_features/aachen

docker exec xfeat_exportador chmod -R a+rwX /workspace/_docker_bridge_aachen/output
cp -r "$BRIDGE/output/"* "$REPO/output/"
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_aachen

echo "=========================================================="
echo " AACHEN DAY-NIGHT CONCLUÍDO EM: $(date)"
echo " Arquivos prontos para submissão em:"
echo "  - $REPO/output/aachen_fp32_orig/Aachen_v1.0_hloc.txt"
echo "  - $REPO/output/aachen_fp32_384x288/Aachen_v1.0_hloc.txt"
echo "  - $REPO/output/aachen_rv1103_int8/Aachen_v1.0_hloc.txt"
echo "=========================================================="
