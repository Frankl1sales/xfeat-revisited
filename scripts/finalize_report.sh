#!/usr/bin/env bash
while tmux has-session -t scannet_board 2>/dev/null || tmux has-session -t scannet_bench 2>/dev/null; do
  sleep 15
done
B="/media/frank/01DBA5D4E1C670D06/Feature_Matching"
R="$B/xfeat-revisited"
D="$B/luckfox-xfeat-quantization/_docker_bridge_final"
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_final
mkdir -p "$D/datasets" "$D/output"
cp -r "$R/assets" "$R/scripts" "$D/"
cp -al "$R/datasets/ScanNet1500" "$R/datasets/hpatches-sequences-release" "$R/datasets/board_features" "$D/datasets/" 2>/dev/null || true
cp -r "$R/output/"* "$D/output/"
docker exec -w /workspace/_docker_bridge_final xfeat_exportador /usr/bin/python3 scripts/eval_board_all.py | tee "$R/output/final_report.log"
docker exec xfeat_exportador chmod -R a+rwX /workspace/_docker_bridge_final/output
cp -r "$D/output/"* "$R/output/"
docker exec xfeat_exportador rm -rf /workspace/_docker_bridge_final
