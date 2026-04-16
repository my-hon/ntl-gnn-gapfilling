#!/bin/bash
# ================================================================
# 脚本6: 方案C v2 增强训练（更大batch、更多层、R²指标、绘图）
#
# 用法:
#   bash deploy/6_train_scheme_c_v2.sh              # 完整流程
#   bash deploy/6_train_scheme_c_v2.sh --skip-build # 跳过图生成
#   bash deploy/6_train_scheme_c_v2.sh --build-only # 只生成图
# ================================================================
set -e

eval "$(conda shell.bash hook)"
conda activate ntl

WORKDIR="/home/featurize/app/ntl-work"
PROJECT_DIR="$WORKDIR/ntl-gnn-gapfilling"
DATA_DIR="$WORKDIR/data"
OUTPUT_BASE="$WORKDIR/scheme_c_v2_train"

NTL_DATA="$DATA_DIR/beijing_2020_ntl.npy"
QUALITY_DATA="$DATA_DIR/beijing_2020_quality.npy"

if [ ! -f "$NTL_DATA" ]; then
    echo "错误: 未找到数据文件 $NTL_DATA"
    exit 1
fi

echo "========================================"
echo "  方案C v2 增强训练"
echo "========================================"
echo "数据: $NTL_DATA"
echo "质量: $QUALITY_DATA"
echo "输出: $OUTPUT_BASE"
echo ""

cd "$PROJECT_DIR"
git pull

python "$PROJECT_DIR/adaptive_graph/scheme_c_v2/train_scheme_c_v2.py" \
    --input "$NTL_DATA" \
    --quality "$QUALITY_DATA" \
    --output "$OUTPUT_BASE" \
    "$@"
