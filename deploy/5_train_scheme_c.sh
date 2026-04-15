#!/bin/bash
# ================================================================
# 脚本5: 方案C 完整训练流程
# 在 Featurize 服务器上运行（需先完成脚本1和脚本2）
#
# 用法:
#   bash deploy/5_train_scheme_c.sh              # 完整流程（生成图+训练）
#   bash deploy/5_train_scheme_c.sh --skip-build # 跳过图生成，直接训练
#   bash deploy/5_train_scheme_c.sh --build-only # 只生成图，不训练
# ================================================================
set -e

eval "$(conda shell.bash hook)"
conda activate ntl

WORKDIR="/home/featurize/app/ntl-work"
PROJECT_DIR="$WORKDIR/ntl-gnn-gapfilling"
DATA_DIR="$WORKDIR/data"
OUTPUT_BASE="$WORKDIR/scheme_c_train"

NTL_DATA="$DATA_DIR/beijing_2020_ntl.npy"
QUALITY_DATA="$DATA_DIR/beijing_2020_quality.npy"

if [ ! -f "$NTL_DATA" ]; then
    echo "错误: 未找到数据文件 $NTL_DATA"
    exit 1
fi

echo "========================================"
echo "  方案C 完整训练流程"
echo "========================================"
echo "数据: $NTL_DATA"
echo "质量: $QUALITY_DATA"
echo "输出: $OUTPUT_BASE"
echo ""

cd "$PROJECT_DIR"
git pull

python "$PROJECT_DIR/adaptive_graph/方案C_attention_graph_learning/train_scheme_c.py" \
    --input "$NTL_DATA" \
    --quality "$QUALITY_DATA" \
    --output "$OUTPUT_BASE" \
    "$@"
