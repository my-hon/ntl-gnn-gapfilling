#!/bin/bash
# ================================================================
# NTL-GNN 全方案自动化运行脚本
# ================================================================
# 功能：
#   1. 自动探测 CUDA 版本
#   2. 安装匹配的 PyTorch + PyG + 依赖
#   3. 生成测试数据（如无真实数据）
#   4. 运行 v1、v2、方案A/B/C 并输出性能对比
#
# 使用方式：
#   bash run_all.sh [--data /path/to/data.npy] [--skip-install]
#
# 依赖：
#   - Python 3.10+ (推荐 3.12)
#   - CUDA 驱动已安装
#   - pip 可用
# ================================================================

set -euo pipefail

# ---- 颜色输出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
header()  { echo -e "\n${CYAN}========================================${NC}"; echo -e "${CYAN}  $*${NC}"; echo -e "${CYAN}========================================${NC}"; }

# ---- 参数解析 ----
SKIP_INSTALL=false
DATA_PATH=""

for arg in "$@"; do
    case $arg in
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        *)
            warn "未知参数: $arg"
            ;;
    esac
done

# ---- 项目根目录 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ================================================================
# 第一步：环境探测
# ================================================================
header "第一步：环境探测"

# 检测 Python
PYTHON=""
for cmd in python3 python; do
    if command -v $cmd &>/dev/null; then
        PYTHON_VERSION=$($cmd --version 2>&1 | grep -oP '\d+\.\d+')
        PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
        PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
            PYTHON=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    error "未找到 Python 3.10+，请先安装 Python"
    exit 1
fi
success "Python: $($PYTHON --version)"

# 检测 CUDA
CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version 2>&1 | grep -oP 'release \K[0-9]+\.[0-9]+')
    success "CUDA (nvcc): $CUDA_VERSION"
elif command -v nvidia-smi &>/dev/null; then
    # 从驱动版本推断 CUDA 版本
    DRIVER_VERSION=$(nvidia-smi | grep -oP 'Driver Version: \K[0-9]+\.[0-9]+')
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
    success "CUDA (nvidia-smi): $CUDA_VERSION (Driver: $DRIVER_VERSION)"
else
    warn "未检测到 CUDA，将安装 CPU 版本 PyTorch"
fi

# 检测 pip
if ! $PYTHON -m pip --version &>/dev/null; then
    error "pip 不可用，请先安装 pip"
    exit 1
fi
success "pip: $($PYTHON -m pip --version)"

# ================================================================
# 第二步：安装依赖
# ================================================================
if [ "$SKIP_INSTALL" = true ]; then
    header "第二步：跳过安装（--skip-install）"
else
    header "第二步：安装 PyTorch + PyG + 依赖"

    # ---- 确定 PyTorch CUDA 版本 ----
    TORCH_CUDA=""
    PYG_CUDA=""

    if [ -n "$CUDA_VERSION" ]; then
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

        # CUDA 版本映射策略：
        #   CUDA 12.x (x<=3) -> cu121 (向前兼容)
        #   CUDA 12.x (x>=4) -> cu124
        #   CUDA 11.x       -> cu118
        if [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -le 3 ]; then
            TORCH_CUDA="cu121"
            PYG_CUDA="cu121"
            info "检测到 CUDA $CUDA_VERSION，使用 PyTorch cu121 构建（向前兼容）"
        elif [ "$CUDA_MAJOR" -eq 12 ]; then
            TORCH_CUDA="cu124"
            PYG_CUDA="cu124"
            info "检测到 CUDA $CUDA_VERSION，使用 PyTorch cu124 构建"
        elif [ "$CUDA_MAJOR" -eq 11 ]; then
            TORCH_CUDA="cu118"
            PYG_CUDA="cu118"
            info "检测到 CUDA $CUDA_VERSION，使用 PyTorch cu118 构建"
        else
            TORCH_CUDA="cu121"
            PYG_CUDA="cu121"
            warn "未知的 CUDA $CUDA_VERSION，默认使用 cu121"
        fi
    fi

    # ---- 安装 PyTorch ----
    info "安装 PyTorch..."
    if [ -n "$TORCH_CUDA" ]; then
        $PYTHON -m pip install torch torchvision torchaudio \
            --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
            --quiet 2>&1 | tail -3
    else
        $PYTHON -m pip install torch torchvision torchaudio --quiet 2>&1 | tail -3
    fi
    success "PyTorch 安装完成"

    # ---- 获取已安装的 PyTorch 版本 ----
    TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__.split('+')[0])")
    TORCH_SERIES=$($PYTHON -c "import torch; v=torch.__version__.split('+')[0]; print('.'.join(v.split('.')[:2]))")
    info "PyTorch 版本: $TORCH_VER (系列: $TORCH_SERIES)"

    # ---- 安装 PyG ----
    info "安装 PyTorch Geometric..."
    $PYTHON -m pip install torch_geometric --quiet 2>&1 | tail -3

    # 安装 PyG 扩展库（匹配 CUDA 版本）
    if [ -n "$PYG_CUDA" ]; then
        $PYTHON -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
            -f "https://data.pyg.org/whl/torch-${TORCH_SERIES}+${PYG_CUDA}.html" \
            --quiet 2>&1 | tail -3 || warn "PyG 扩展库安装部分失败（核心功能仍可用）"
    else
        $PYTHON -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
            --quiet 2>&1 | tail -3 || warn "PyG 扩展库安装部分失败（核心功能仍可用）"
    fi
    success "PyG 安装完成"

    # ---- 安装其他依赖 ----
    info "安装其他依赖（numpy, scipy, numba, tqdm）..."
    $PYTHON -m pip install numpy scipy numba tqdm --quiet 2>&1 | tail -3
    success "依赖安装完成"
fi

# ---- 验证安装 ----
header "验证安装"
echo ""
$PYTHON -c "
import torch
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA 可用:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA 版本:     {torch.version.cuda}')
    print(f'  GPU:           {torch.cuda.get_device_name(0)}')
    print(f'  GPU 显存:      {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

try:
    import torch_geometric
    print(f'  PyG:           {torch_geometric.__version__}')
except ImportError:
    print('  PyG:           未安装')

try:
    import numba
    print(f'  Numba:         {numba.__version__}')
except ImportError:
    print('  Numba:         未安装')

import numpy as np
print(f'  NumPy:         {np.__version__}')
"

# ================================================================
# 第三步：准备测试数据
# ================================================================
header "第三步：准备测试数据"

if [ -n "$DATA_PATH" ] && [ -f "$DATA_PATH" ]; then
    info "使用指定数据: $DATA_PATH"
    TEST_DATA="$DATA_PATH"
else
    # 生成模拟数据（北京 2020 Black Marble 格式）
    TEST_DATA="$SCRIPT_DIR/test_data.npy"
    info "未指定真实数据，生成模拟测试数据: $TEST_DATA"

    $PYTHON -c "
import numpy as np
np.random.seed(42)

# 模拟北京 2020 Black Marble 数据: (366天, 476行, 520列)
T, H, W = 366, 476, 520
data = np.random.exponential(scale=2.0, size=(T, H, W)).astype(np.float32)

# 添加空间模式（城市中心更亮）
cy, cx = H // 2, W // 2
y_coords = np.arange(H).reshape(-1, 1)
x_coords = np.arange(W).reshape(1, -1)
spatial_pattern = np.exp(-((y_coords - cy)**2 + (x_coords - cx)**2) / (2 * 150**2))
data *= spatial_pattern[np.newaxis, :, :]

# 添加时间模式（冬季灯光更亮）
time_pattern = 1.0 + 0.3 * np.cos(2 * np.pi * np.arange(T) / 365 - np.pi / 2)
data *= time_pattern[:, np.newaxis, np.newaxis]

# 添加缺失值（约 30% 云层遮挡）
mask = np.random.random((T, H, W)) < 0.30
data[mask] = np.nan

np.save('$TEST_DATA', data)
print(f'  测试数据已生成: shape={data.shape}, 缺失率={mask.mean():.1%}')
"
    success "测试数据生成完成"
fi

# ================================================================
# 第四步：运行各方案
# ================================================================

# ---- 通用参数 ----
WORKERS=$(nproc)
if [ "$WORKERS" -gt 8 ]; then
    WORKERS=8
fi
OUTPUT_BASE="$SCRIPT_DIR/run_results"
mkdir -p "$OUTPUT_BASE"

# ---- 结果收集文件 ----
RESULTS_FILE="$OUTPUT_BASE/benchmark_results.txt"
echo "NTL-GNN 全方案性能对比" > "$RESULTS_FILE"
echo "运行时间: $(date)" >> "$RESULTS_FILE"
echo "数据: $TEST_DATA" >> "$RESULTS_FILE"
echo "Python: $($PYTHON --version)" >> "$RESULTS_FILE"
echo "PyTorch: $($PYTHON -c 'import torch; print(torch.__version__)')" >> "$RESULTS_FILE"
echo "CUDA: $($PYTHON -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else \"N/A\")')" >> "$RESULTS_FILE"
echo "GPU: $($PYTHON -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")')" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# ================================================================
# 方案 1: v1 (GPU Bresenham - 基线参考)
# ================================================================
header "方案 1/5: v1 - GPU Bresenham（基线参考）"

V1_OUTPUT="$OUTPUT_BASE/v1_output"
mkdir -p "$V1_OUTPUT"

$PYTHON -c "
import sys, time, os
sys.path.insert(0, '$SCRIPT_DIR')

try:
    from ntl_graph_accel.config import Config
    from ntl_graph_accel.data_loader import NTLDataLoader
    from ntl_graph_accel.spatial_partitioner import ParallelGraphProcessor

    config = Config()
    config.input_path = '$TEST_DATA'
    config.output_dir = '$V1_OUTPUT'
    config.cache_dir = '$OUTPUT_BASE/v1_cache'
    config.accel.num_workers = $WORKERS
    config.accel.use_cuda = False  # v1 GPU Bresenham 已验证无效，使用 CPU
    config.accel.use_cache = False  # 禁用缓存以测试原始性能

    # 加载数据
    loader = NTLDataLoader(config)
    loader.load(path=config.input_path)
    positions = loader.get_missing_positions()

    # 采样测试（取 1000 个位置进行基准测试）
    import numpy as np
    rng = np.random.RandomState(0)
    n_test = min(1000, len(positions))
    test_pos = positions[rng.choice(len(positions), size=n_test, replace=False)]

    # 构建
    processor = ParallelGraphProcessor(config)
    t0 = time.time()
    graphs = processor.process(loader.data, loader.valid_mask, test_pos, mode='missing')
    elapsed = time.time() - t0

    speed = len(graphs) / max(elapsed, 0.001)
    print(f'  v1 结果: {len(graphs)} 图, 耗时 {elapsed:.2f}s, 速度 {speed:.0f} 图/秒')

    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[v1 GPU Bresenham]\n')
        f.write(f'  构建图数: {len(graphs)}\n')
        f.write(f'  耗时: {elapsed:.2f}s\n')
        f.write(f'  速度: {speed:.0f} 图/秒\n\n')

except Exception as e:
    print(f'  v1 运行失败: {e}')
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[v1 GPU Bresenham] 运行失败: {e}\n\n')
" 2>&1 | grep -v "^$" || warn "v1 运行异常"

# ================================================================
# 方案 2: v2 (Numba JIT)
# ================================================================
header "方案 2/5: v2 - Numba JIT 加速"

V2_OUTPUT="$OUTPUT_BASE/v2_output"
mkdir -p "$V2_OUTPUT"

$PYTHON -c "
import sys, time, os
sys.path.insert(0, '$SCRIPT_DIR')

try:
    from ntl_graph_accel_v2.config import Config
    from ntl_graph_accel_v2.data_loader import NTLDataLoader
    from ntl_graph_accel_v2.spatial_partitioner import ParallelGraphProcessor

    config = Config()
    config.input_path = '$TEST_DATA'
    config.output_dir = '$V2_OUTPUT'
    config.cache_dir = '$OUTPUT_BASE/v2_cache'
    config.accel.num_workers = $WORKERS
    config.accel.use_numba = True
    config.accel.use_cache = False  # 禁用缓存以测试原始性能

    # 加载数据
    loader = NTLDataLoader(config)
    loader.load(path=config.input_path)
    positions = loader.get_missing_positions()

    # 采样测试
    import numpy as np
    rng = np.random.RandomState(0)
    n_test = min(1000, len(positions))
    test_pos = positions[rng.choice(len(positions), size=n_test, replace=False)]

    # 构建
    processor = ParallelGraphProcessor(config)
    t0 = time.time()
    graphs = processor.process(loader.data, loader.valid_mask, test_pos, mode='missing')
    elapsed = time.time() - t0

    speed = len(graphs) / max(elapsed, 0.001)
    print(f'  v2 结果: {len(graphs)} 图, 耗时 {elapsed:.2f}s, 速度 {speed:.0f} 图/秒')

    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[v2 Numba JIT]\n')
        f.write(f'  构建图数: {len(graphs)}\n')
        f.write(f'  耗时: {elapsed:.2f}s\n')
        f.write(f'  速度: {speed:.0f} 图/秒\n\n')

except Exception as e:
    print(f'  v2 运行失败: {e}')
    import traceback
    traceback.print_exc()
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[v2 Numba JIT] 运行失败: {e}\n\n')
" 2>&1 | grep -v "^$" || warn "v2 运行异常"

# ================================================================
# 方案 3: 方案A - 质量驱动自适应节点数
# ================================================================
header "方案 3/5: 方案A - 质量驱动自适应节点数"

A_OUTPUT="$OUTPUT_BASE/scheme_a_output"
mkdir -p "$A_OUTPUT"

$PYTHON -c "
import sys, time, os
sys.path.insert(0, '$SCRIPT_DIR')

try:
    import numpy as np
    from adaptive_graph.scheme_a_quality_adaptive_nodes.config_a import ConfigA
    from adaptive_graph.scheme_a_quality_adaptive_nodes.graph_builder_a import GraphBuilderA

    config = ConfigA()
    config.data.data_shape = (366, 476, 520)
    config.input_path = '$TEST_DATA'
    config.output_dir = '$A_OUTPUT'
    config.cache_dir = '$OUTPUT_BASE/scheme_a_cache'

    # 加载数据
    data = np.load('$TEST_DATA')
    valid_mask = ~np.isnan(data)

    # 创建构建器
    builder = GraphBuilderA(config, data, valid_mask)

    # 预计算质量图
    print('  预计算质量图...')
    t_quality = time.time()
    builder.precompute_quality()
    quality_time = time.time() - t_quality
    print(f'  质量图计算耗时: {quality_time:.2f}s')

    # 获取缺失位置
    missing_t, missing_h, missing_w = np.where(np.isnan(data))
    positions = np.column_stack([missing_t, missing_h, missing_w])

    # 采样测试
    rng = np.random.RandomState(0)
    n_test = min(1000, len(positions))
    test_pos = positions[rng.choice(len(positions), size=n_test, replace=False)]

    # 构建子图
    print(f'  构建 {n_test} 个子图...')
    t0 = time.time()
    graphs = builder.build_batch(test_pos)
    elapsed = time.time() - t0

    speed = len(graphs) / max(elapsed, 0.001)

    # 统计自适应节点数分布
    node_counts = [g.num_nodes for g in graphs]
    print(f'  方案A 结果: {len(graphs)} 图, 耗时 {elapsed:.2f}s, 速度 {speed:.0f} 图/秒')
    print(f'  节点数分布: min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}, std={np.std(node_counts):.1f}')

    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案A 质量自适应节点]\n')
        f.write(f'  构建图数: {len(graphs)}\n')
        f.write(f'  质量图计算: {quality_time:.2f}s\n')
        f.write(f'  子图构建: {elapsed:.2f}s\n')
        f.write(f'  速度: {speed:.0f} 图/秒\n')
        f.write(f'  节点数: min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}\n\n')

except Exception as e:
    print(f'  方案A 运行失败: {e}')
    import traceback
    traceback.print_exc()
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案A 质量自适应节点] 运行失败: {e}\n\n')
" 2>&1 | grep -v "^$" || warn "方案A 运行异常"

# ================================================================
# 方案 4: 方案B - 空间异质性感知动态连边
# ================================================================
header "方案 4/5: 方案B - 空间异质性感知动态连边"

B_OUTPUT="$OUTPUT_BASE/scheme_b_output"
mkdir -p "$B_OUTPUT"

$PYTHON -c "
import sys, time, os
sys.path.insert(0, '$SCRIPT_DIR')

try:
    import numpy as np
    from adaptive_graph.scheme_b_dynamic_edge_construction.config_b import ConfigB
    from adaptive_graph.scheme_b_dynamic_edge_construction.graph_builder_b import GraphBuilderB

    config = ConfigB()
    config.data.data_shape = (366, 476, 520)
    config.input_path = '$TEST_DATA'
    config.output_dir = '$B_OUTPUT'
    config.cache_dir = '$OUTPUT_BASE/scheme_b_cache'

    # 加载数据
    data = np.load('$TEST_DATA')
    valid_mask = ~np.isnan(data)

    # 创建构建器
    builder = GraphBuilderB(config, data, valid_mask)

    # 获取缺失位置
    missing_t, missing_h, missing_w = np.where(np.isnan(data))
    positions = np.column_stack([missing_t, missing_h, missing_w])

    # 采样测试
    rng = np.random.RandomState(0)
    n_test = min(1000, len(positions))
    test_pos = positions[rng.choice(len(positions), size=n_test, replace=False)]

    # 构建子图
    print(f'  构建 {n_test} 个子图...')
    t0 = time.time()
    graphs = builder.build_batch(test_pos)
    elapsed = time.time() - t0

    speed = len(graphs) / max(elapsed, 0.001)

    # 统计
    node_counts = [g.num_nodes for g in graphs]
    edge_counts = [g.edge_index.shape[1] if hasattr(g, 'edge_index') and g.edge_index is not None else 0 for g in graphs]
    het_indices = [g.heterogeneity_index for g in graphs if hasattr(g, 'heterogeneity_index')]

    print(f'  方案B 结果: {len(graphs)} 图, 耗时 {elapsed:.2f}s, 速度 {speed:.0f} 图/秒')
    print(f'  节点数: mean={np.mean(node_counts):.1f}')
    print(f'  边数: mean={np.mean(edge_counts):.1f}')
    if het_indices:
        print(f'  异质性指数: mean={np.mean(het_indices):.4f}, std={np.std(het_indices):.4f}')

    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案B 异质性动态连边]\n')
        f.write(f'  构建图数: {len(graphs)}\n')
        f.write(f'  耗时: {elapsed:.2f}s\n')
        f.write(f'  速度: {speed:.0f} 图/秒\n')
        f.write(f'  节点数: mean={np.mean(node_counts):.1f}\n')
        f.write(f'  边数: mean={np.mean(edge_counts):.1f}\n')
        if het_indices:
            f.write(f'  异质性指数: mean={np.mean(het_indices):.4f}\n\n')
        else:
            f.write('\n')

except Exception as e:
    print(f'  方案B 运行失败: {e}')
    import traceback
    traceback.print_exc()
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案B 异质性动态连边] 运行失败: {e}\n\n')
" 2>&1 | grep -v "^$" || warn "方案B 运行异常"

# ================================================================
# 方案 5: 方案C - 注意力图结构学习
# ================================================================
header "方案 5/5: 方案C - 注意力图结构学习（需要 PyTorch）"

C_OUTPUT="$OUTPUT_BASE/scheme_c_output"
mkdir -p "$C_OUTPUT"

$PYTHON -c "
import sys, time, os
sys.path.insert(0, '$SCRIPT_DIR')

try:
    import torch
    import numpy as np
    from adaptive_graph.scheme_c_attention_graph_learning.config_c import ConfigC
    from adaptive_graph.scheme_c_attention_graph_learning.graph_builder_c import GraphBuilderC

    config = ConfigC()
    config.data.data_shape = (366, 476, 520)
    config.input_path = '$TEST_DATA'
    config.output_dir = '$C_OUTPUT'
    config.cache_dir = '$OUTPUT_BASE/scheme_c_cache'

    device = config.get_device()
    print(f'  使用设备: {device}')

    # 加载数据
    data = np.load('$TEST_DATA')
    valid_mask = ~np.isnan(data)

    # 创建构建器
    builder = GraphBuilderC(config, data, valid_mask)

    # 获取缺失位置
    missing_t, missing_h, missing_w = np.where(np.isnan(data))
    positions = np.column_stack([missing_t, missing_h, missing_w])

    # 采样测试（方案C 较慢，取 500 个）
    rng = np.random.RandomState(0)
    n_test = min(500, len(positions))
    test_pos = positions[rng.choice(len(positions), size=n_test, replace=False)]

    # 构建子图
    print(f'  构建 {n_test} 个子图（含 MLP 节点选择）...')
    t0 = time.time()
    graphs = builder.build_batch(test_pos)
    elapsed = time.time() - t0

    speed = len(graphs) / max(elapsed, 0.001)

    # 统计
    node_counts = [g.num_nodes for g in graphs]
    print(f'  方案C 结果: {len(graphs)} 图, 耗时 {elapsed:.2f}s, 速度 {speed:.0f} 图/秒')
    print(f'  节点数: mean={np.mean(node_counts):.1f}')

    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案C 注意力图结构学习]\n')
        f.write(f'  构建图数: {len(graphs)}\n')
        f.write(f'  耗时: {elapsed:.2f}s\n')
        f.write(f'  速度: {speed:.0f} 图/秒\n')
        f.write(f'  节点数: mean={np.mean(node_counts):.1f}\n\n')

    # ---- 可选：运行 GNN 训练演示 ----
    print()
    print('  ---- GNN 训练演示（10 个 epoch）----')
    try:
        from adaptive_graph.scheme_c_attention_graph_learning.model_c import (
            AttentionGraphModel, Trainer, GraphDatasetC
        )

        model_config = config.gnn_model
        model_config.num_epochs = 10  # 演示用，仅训练 10 轮

        model = AttentionGraphModel(config).to(device)
        trainer = Trainer(config, model)

        # 创建简单训练数据
        if len(graphs) > 0:
            train_size = int(len(graphs) * 0.8)
            train_graphs = graphs[:train_size]
            val_graphs = graphs[train_size:]

            # 生成伪标签（演示用）
            train_labels = np.random.randn(len(train_graphs)).astype(np.float32)
            val_labels = np.random.randn(len(val_graphs)).astype(np.float32)

            train_dataset = GraphDatasetC(train_graphs, train_labels)
            val_dataset = GraphDatasetC(val_graphs, val_labels)

            t_train = time.time()
            history = trainer.train(train_dataset, val_dataset)
            train_elapsed = time.time() - t_train

            print(f'  训练完成: {model_config.num_epochs} epochs, 耗时 {train_elapsed:.2f}s')
            print(f'  最终训练损失: {history[\"train_loss\"][-1]:.4f}')

            with open('$RESULTS_FILE', 'a') as f:
                f.write(f'  GNN 训练: {model_config.num_epochs} epochs, 耗时 {train_elapsed:.2f}s\n')
                f.write(f'  最终训练损失: {history[\"train_loss\"][-1]:.4f}\n\n')
    except Exception as e:
        print(f'  GNN 训练跳过: {e}')

except ImportError as e:
    print(f'  方案C 需要 PyTorch，跳过: {e}')
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案C 注意力图结构学习] 跳过（缺少 PyTorch）: {e}\n\n')
except Exception as e:
    print(f'  方案C 运行失败: {e}')
    import traceback
    traceback.print_exc()
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案C 注意力图结构学习] 运行失败: {e}\n\n')
" 2>&1 | grep -v "^$" || warn "方案C 运行异常"

# ================================================================
# 汇总报告
# ================================================================
header "性能对比汇总"
echo ""
cat "$RESULTS_FILE"
echo ""
info "详细结果已保存至: $RESULTS_FILE"
info "各方案输出目录: $OUTPUT_BASE/"
echo ""
success "全部方案运行完成！"
