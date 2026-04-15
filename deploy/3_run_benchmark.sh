#!/bin/bash
# ================================================================
# 脚本3: 运行全部方案的基准测试
# 在 Featurize 服务器上运行（需先完成脚本1和脚本2）
# ================================================================
set -e

eval "$(conda shell.bash hook)"
conda activate ntl

WORKDIR="/home/featurize/app/ntl-work"
PROJECT_DIR="$WORKDIR/ntl-gnn-gapfilling"
DATA_DIR="$WORKDIR/data"
OUTPUT_BASE="$WORKDIR/run_results"
mkdir -p "$OUTPUT_BASE"

NTL_DATA="$DATA_DIR/beijing_2020_ntl.npy"
QUALITY_DATA="$DATA_DIR/beijing_2020_quality.npy"

if [ ! -f "$NTL_DATA" ]; then
    echo "错误: 未找到数据文件 $NTL_DATA"
    echo "请先运行: bash $PROJECT_DIR/deploy/2_prepare_data.sh"
    exit 1
fi

WORKERS=$(nproc)
if [ "$WORKERS" -gt 8 ]; then
    WORKERS=8
fi

echo "========================================"
echo "  基准测试: v1 + v2 图构建"
echo "========================================"
echo "数据: $NTL_DATA"
echo "质量: $QUALITY_DATA"
echo "Workers: $WORKERS"
echo "输出: $OUTPUT_BASE"
echo ""

RESULTS_FILE="$OUTPUT_BASE/benchmark_results.txt"
echo "NTL-GNN 基准测试" > "$RESULTS_FILE"
echo "运行时间: $(date)" >> "$RESULTS_FILE"
python -c "
import torch, numpy as np
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'NumPy: {np.__version__}')
" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# ================================================================
# 方案 1: v1 (GPU Bresenham - 基线参考)
# ================================================================
echo "========================================"
echo "  [1/2] v1 - 基线参考"
echo "========================================"

V1_OUTPUT="$OUTPUT_BASE/v1_output"

python -c "
import sys, time, os, numpy as np
sys.path.insert(0, '$PROJECT_DIR')

try:
    from ntl_graph_accel.main import run_train
    from ntl_graph_accel.config import Config

    config = Config()
    config.input_path = '$NTL_DATA'
    config.data.quality_path = '$QUALITY_DATA'
    config.output_dir = '$V1_OUTPUT'
    config.graph.sample_per_class = 100
    config.graph.search_node = 32
    config.graph.ext_range = 6
    config.seed = 0

    t0 = time.time()
    run_train(config)
    elapsed = time.time() - t0

    print(f'  v1 总耗时: {elapsed:.2f}s')

    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[v1 基线]\n')
        f.write(f'  总耗时: {elapsed:.2f}s\n\n')

except Exception as e:
    print(f'  v1 运行失败: {e}')
    import traceback; traceback.print_exc()
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[v1 基线] 运行失败: {e}\n\n')
"

# ================================================================
# 方案 2: v2 (Numba JIT 加速)
# ================================================================
echo ""
echo "========================================"
echo "  [2/2] v2 - Numba JIT 加速"
echo "========================================"

V2_OUTPUT="$OUTPUT_BASE/v2_output"

python -c "
import sys, time, os, numpy as np
sys.path.insert(0, '$PROJECT_DIR')

try:
    from ntl_graph_accel_v2.main import run_train
    from ntl_graph_accel_v2.config import Config

    config = Config()
    config.input_path = '$NTL_DATA'
    config.data.quality_path = '$QUALITY_DATA'
    config.output_dir = '$V2_OUTPUT'
    config.graph.sample_per_class = 100
    config.graph.search_node = 32
    config.graph.ext_range = 6
    config.accel.use_numba = True
    config.seed = 0

    t0 = time.time()
    run_train(config)
    elapsed = time.time() - t0

    print(f'  v2 总耗时: {elapsed:.2f}s')

    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[v2 Numba JIT]\n')
        f.write(f'  总耗时: {elapsed:.2f}s\n\n')

except Exception as e:
    print(f'  v2 运行失败: {e}')
    import traceback; traceback.print_exc()
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[v2 Numba JIT] 运行失败: {e}\n\n')
"

# ================================================================
# 汇总
# ================================================================
echo ""
echo "========================================"
echo "  性能对比汇总"
echo "========================================"
echo ""
cat "$RESULTS_FILE"
echo ""
echo "详细结果: $RESULTS_FILE"
echo "各方案输出: $OUTPUT_BASE/"
echo ""
echo "========================================"
echo "  全部完成!"
echo "========================================"
