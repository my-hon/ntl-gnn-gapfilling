#!/bin/bash
# ================================================================
# 脚本2: 准备数据 - 将 TIF 文件转换为 NumPy 数组
# 在 Featurize 服务器上运行（需先完成脚本1）
# ================================================================
set -e

eval "$(conda shell.bash hook)"
conda activate ntl

WORKDIR="/home/featurize/app/ntl-work"
PROJECT_DIR="$WORKDIR/ntl-gnn-gapfilling"
DATA_DIR="/home/featurize/data/output_tif/Beijing"
OUTPUT_DIR="$WORKDIR/data"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  数据准备: TIF -> NumPy"
echo "========================================"

python "$PROJECT_DIR/deploy/prepare_bj_data.py" \
    --input "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --year 2020

echo ""
echo "========================================"
echo "  数据准备完成!"
echo "========================================"
echo "输出目录: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR/"
