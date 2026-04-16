#!/bin/bash
# ================================================================
# 脚本7: 方案A/B 图构建 + 训练
# 在 Featurize 服务器上运行（需先完成脚本1和脚本2）
#
# 用法:
#   bash deploy/7_train_schemes_ab.sh                  # 完整流程（v2+A+B 全部构建+训练）
#   bash deploy/7_train_schemes_ab.sh --skip-build     # 跳过图构建，直接训练
#   bash deploy/7_train_schemes_ab.sh --build-only     # 只构建图，不训练
#   bash deploy/7_train_schemes_ab.sh --scheme scheme_a  # 只运行方案A
#   bash deploy/7_train_schemes_ab.sh --scheme scheme_b  # 只运行方案B
#   bash deploy/7_train_schemes_ab.sh --scheme v2        # 只运行v2基线
# ================================================================
set -e

eval "$(conda shell.bash hook)"
conda activate ntl

WORKDIR="/home/featurize/app/ntl-work"
PROJECT_DIR="$WORKDIR/ntl-gnn-gapfilling"
DATA_DIR="$WORKDIR/data"
OUTPUT_BASE="$WORKDIR/ab_train_results"

NTL_DATA="$DATA_DIR/beijing_2020_ntl.npy"
QUALITY_DATA="$DATA_DIR/beijing_2020_quality.npy"

if [ ! -f "$NTL_DATA" ]; then
    echo "错误: 未找到数据文件 $NTL_DATA"
    exit 1
fi

# 解析参数
SCHEME=""
SKIP_BUILD=false
BUILD_ONLY=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --scheme)
            SCHEME="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

cd "$PROJECT_DIR"
git pull

# 训练参数（更大的模型配置以提高 R²）
TRAIN_ARGS="--hidden-dim 128 --num-heads 8 --num-layers 4 --batch-size 256 --epochs 300 --lr 1e-3 --patience 40"

run_scheme() {
    local scheme=$1
    local name=$2
    local output_dir="$OUTPUT_BASE/$scheme"

    echo ""
    echo "========================================"
    echo "  $name"
    echo "========================================"
    echo "数据: $NTL_DATA"
    echo "质量: $QUALITY_DATA"
    echo "输出: $output_dir"
    echo ""

    local cmd="python $PROJECT_DIR/adaptive_graph/train_baseline/build_and_train_ab.py \
        --scheme $scheme \
        --input $NTL_DATA \
        --quality $QUALITY_DATA \
        --output $output_dir \
        $TRAIN_ARGS $EXTRA_ARGS"

    if [ "$SKIP_BUILD" = true ]; then
        cmd="$cmd --skip-build --graphs $output_dir/graphs/${scheme}_graphs.pkl"
    fi

    if [ "$BUILD_ONLY" = true ]; then
        cmd="$cmd --build-only"
    fi

    echo "执行命令: $cmd"
    eval $cmd
    echo ""
    echo "$name 完成!"
    echo "========================================"
}

# 根据参数决定运行哪些方案
if [ -n "$SCHEME" ]; then
    case $SCHEME in
        v2)
            run_scheme "v2" "v2基线"
            ;;
        scheme_a)
            run_scheme "scheme_a" "方案A(质量自适应)"
            ;;
        scheme_b)
            run_scheme "scheme_b" "方案B(异质性感知)"
            ;;
        *)
            echo "错误: 未知方案 $SCHEME (可选: v2, scheme_a, scheme_b)"
            exit 1
            ;;
    esac
else
    # 默认: 依次运行 v2, 方案A, 方案B
    echo "将依次运行: v2基线 → 方案A → 方案B"
    echo ""

    run_scheme "v2" "v2基线"
    run_scheme "scheme_a" "方案A(质量自适应)"
    run_scheme "scheme_b" "方案B(异质性感知)"
fi

echo ""
echo "========================================"
echo "  全部完成!"
echo "  结果目录: $OUTPUT_BASE"
echo "========================================"
