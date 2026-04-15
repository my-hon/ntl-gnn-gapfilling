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

# 数据文件
NTL_DATA="$DATA_DIR/beijing_2020_ntl.npy"
QUALITY_DATA="$DATA_DIR/beijing_2020_quality.npy"
CLOUD_DATA="$DATA_DIR/beijing_2020_cloud.npy"

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
echo "  基准测试: 全部方案"
echo "========================================"
echo "数据: $NTL_DATA"
echo "Workers: $WORKERS"
echo "输出: $OUTPUT_BASE"
echo ""

RESULTS_FILE="$OUTPUT_BASE/benchmark_results.txt"
echo "NTL-GNN 全方案性能对比" > "$RESULTS_FILE"
echo "运行时间: $(date)" >> "$RESULTS_FILE"
echo "数据: $NTL_DATA" >> "$RESULTS_FILE"
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
echo "  [1/5] v1 - GPU Bresenham（基线参考）"
echo "========================================"

V1_OUTPUT="$OUTPUT_BASE/v1_output"
mkdir -p "$V1_OUTPUT"

python -c "
import sys, time, os
sys.path.insert(0, '$PROJECT_DIR')

try:
    import numpy as np
    from ntl_graph_accel.config import Config
    from ntl_graph_accel.data_loader import NTLDataLoader
    from ntl_graph_accel.spatial_partitioner import ParallelGraphProcessor

    config = Config()
    config.input_path = '$NTL_DATA'
    config.output_dir = '$V1_OUTPUT'
    config.cache_dir = '$OUTPUT_BASE/v1_cache'
    config.accel.num_workers = $WORKERS
    config.accel.use_cuda = False
    config.accel.use_cache = False

    loader = NTLDataLoader(config)
    loader.load(path=config.input_path)
    positions = loader.get_missing_positions()

    rng = np.random.RandomState(0)
    n_test = min(1000, len(positions))
    test_pos = positions[rng.choice(len(positions), size=n_test, replace=False)]

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
    import traceback; traceback.print_exc()
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[v1 GPU Bresenham] 运行失败: {e}\n\n')
"

# ================================================================
# 方案 2: v2 (Numba JIT)
# ================================================================
echo ""
echo "========================================"
echo "  [2/5] v2 - Numba JIT 加速"
echo "========================================"

V2_OUTPUT="$OUTPUT_BASE/v2_output"
mkdir -p "$V2_OUTPUT"

python -c "
import sys, time, os
sys.path.insert(0, '$PROJECT_DIR')

try:
    import numpy as np
    from ntl_graph_accel_v2.config import Config
    from ntl_graph_accel_v2.data_loader import NTLDataLoader
    from ntl_graph_accel_v2.spatial_partitioner import ParallelGraphProcessor

    config = Config()
    config.input_path = '$NTL_DATA'
    config.output_dir = '$V2_OUTPUT'
    config.cache_dir = '$OUTPUT_BASE/v2_cache'
    config.accel.num_workers = $WORKERS
    config.accel.use_numba = True
    config.accel.use_cache = False

    loader = NTLDataLoader(config)
    loader.load(path=config.input_path)
    positions = loader.get_missing_positions()

    rng = np.random.RandomState(0)
    n_test = min(1000, len(positions))
    test_pos = positions[rng.choice(len(positions), size=n_test, replace=False)]

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
    import traceback; traceback.print_exc()
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[v2 Numba JIT] 运行失败: {e}\n\n')
"

# ================================================================
# 方案 3: 方案A - 质量驱动自适应节点数
# ================================================================
echo ""
echo "========================================"
echo "  [3/5] 方案A - 质量驱动自适应节点数"
echo "========================================"

A_OUTPUT="$OUTPUT_BASE/scheme_a_output"
mkdir -p "$A_OUTPUT"

python -c "
import sys, time, os
sys.path.insert(0, '$PROJECT_DIR')

try:
    import numpy as np
    from adaptive_graph.方案A_quality_adaptive_nodes.config_a import ConfigA
    from adaptive_graph.方案A_quality_adaptive_nodes.graph_builder_a import GraphBuilderA

    config = ConfigA()
    data = np.load('$NTL_DATA')
    config.data.data_shape = data.shape
    config.output_dir = '$A_OUTPUT'
    config.cache_dir = '$OUTPUT_BASE/scheme_a_cache'

    valid_mask = ~np.isnan(data)
    builder = GraphBuilderA(config, data, valid_mask)

    print('  预计算质量图...')
    t_quality = time.time()
    builder.precompute_quality()
    quality_time = time.time() - t_quality
    print(f'  质量图计算耗时: {quality_time:.2f}s')

    missing_t, missing_h, missing_w = np.where(np.isnan(data))
    positions = np.column_stack([missing_t, missing_h, missing_w])

    rng = np.random.RandomState(0)
    n_test = min(1000, len(positions))
    test_pos = positions[rng.choice(len(positions), size=n_test, replace=False)]

    print(f'  构建 {n_test} 个子图...')
    t0 = time.time()
    graphs = builder.build_batch(test_pos)
    elapsed = time.time() - t0

    speed = len(graphs) / max(elapsed, 0.001)
    node_counts = [g.num_nodes for g in graphs]
    print(f'  方案A 结果: {len(graphs)} 图, 耗时 {elapsed:.2f}s, 速度 {speed:.0f} 图/秒')
    print(f'  节点数: min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}')

    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案A 质量自适应节点]\n')
        f.write(f'  构建图数: {len(graphs)}\n')
        f.write(f'  质量图计算: {quality_time:.2f}s\n')
        f.write(f'  子图构建: {elapsed:.2f}s\n')
        f.write(f'  速度: {speed:.0f} 图/秒\n')
        f.write(f'  节点数: min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}\n\n')

except Exception as e:
    print(f'  方案A 运行失败: {e}')
    import traceback; traceback.print_exc()
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案A 质量自适应节点] 运行失败: {e}\n\n')
"

# ================================================================
# 方案 4: 方案B - 空间异质性感知动态连边
# ================================================================
echo ""
echo "========================================"
echo "  [4/5] 方案B - 空间异质性感知动态连边"
echo "========================================"

B_OUTPUT="$OUTPUT_BASE/scheme_b_output"
mkdir -p "$B_OUTPUT"

python -c "
import sys, time, os
sys.path.insert(0, '$PROJECT_DIR')

try:
    import numpy as np
    from adaptive_graph.方案B_dynamic_edge_construction.config_b import ConfigB
    from adaptive_graph.方案B_dynamic_edge_construction.graph_builder_b import GraphBuilderB

    config = ConfigB()
    data = np.load('$NTL_DATA')
    config.data.data_shape = data.shape
    config.output_dir = '$B_OUTPUT'
    config.cache_dir = '$OUTPUT_BASE/scheme_b_cache'

    valid_mask = ~np.isnan(data)
    builder = GraphBuilderB(config, data, valid_mask)

    missing_t, missing_h, missing_w = np.where(np.isnan(data))
    positions = np.column_stack([missing_t, missing_h, missing_w])

    rng = np.random.RandomState(0)
    n_test = min(1000, len(positions))
    test_pos = positions[rng.choice(len(positions), size=n_test, replace=False)]

    print(f'  构建 {n_test} 个子图...')
    t0 = time.time()
    graphs = builder.build_batch(test_pos)
    elapsed = time.time() - t0

    speed = len(graphs) / max(elapsed, 0.001)
    node_counts = [g.num_nodes for g in graphs]
    edge_counts = [g.edge_index.shape[1] if hasattr(g, 'edge_index') and g.edge_index is not None else 0 for g in graphs]
    het_indices = [g.heterogeneity_index for g in graphs if hasattr(g, 'heterogeneity_index')]

    print(f'  方案B 结果: {len(graphs)} 图, 耗时 {elapsed:.2f}s, 速度 {speed:.0f} 图/秒')
    print(f'  节点数: mean={np.mean(node_counts):.1f}, 边数: mean={np.mean(edge_counts):.1f}')
    if het_indices:
        print(f'  异质性指数: mean={np.mean(het_indices):.4f}')

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
    import traceback; traceback.print_exc()
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案B 异质性动态连边] 运行失败: {e}\n\n')
"

# ================================================================
# 方案 5: 方案C - 注意力图结构学习
# ================================================================
echo ""
echo "========================================"
echo "  [5/5] 方案C - 注意力图结构学习"
echo "========================================"

C_OUTPUT="$OUTPUT_BASE/scheme_c_output"
mkdir -p "$C_OUTPUT"

python -c "
import sys, time, os
sys.path.insert(0, '$PROJECT_DIR')

try:
    import torch
    import numpy as np
    from adaptive_graph.方案C_attention_graph_learning.config_c import ConfigC
    from adaptive_graph.方案C_attention_graph_learning.graph_builder_c import GraphBuilderC

    config = ConfigC()
    data = np.load('$NTL_DATA')
    config.data.data_shape = data.shape
    config.output_dir = '$C_OUTPUT'
    config.cache_dir = '$OUTPUT_BASE/scheme_c_cache'

    device = config.get_device()
    print(f'  使用设备: {device}')

    valid_mask = ~np.isnan(data)
    builder = GraphBuilderC(config, data, valid_mask)

    missing_t, missing_h, missing_w = np.where(np.isnan(data))
    positions = np.column_stack([missing_t, missing_h, missing_w])

    rng = np.random.RandomState(0)
    n_test = min(500, len(positions))
    test_pos = positions[rng.choice(len(positions), size=n_test, replace=False)]

    print(f'  构建 {n_test} 个子图...')
    t0 = time.time()
    graphs = builder.build_batch(test_pos)
    elapsed = time.time() - t0

    speed = len(graphs) / max(elapsed, 0.001)
    node_counts = [g.num_nodes for g in graphs]
    print(f'  方案C 结果: {len(graphs)} 图, 耗时 {elapsed:.2f}s, 速度 {speed:.0f} 图/秒')
    print(f'  节点数: mean={np.mean(node_counts):.1f}')

    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案C 注意力图结构学习]\n')
        f.write(f'  构建图数: {len(graphs)}\n')
        f.write(f'  耗时: {elapsed:.2f}s\n')
        f.write(f'  速度: {speed:.0f} 图/秒\n')
        f.write(f'  节点数: mean={np.mean(node_counts):.1f}\n\n')

except ImportError as e:
    print(f'  方案C 需要 PyTorch，跳过: {e}')
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案C 注意力图结构学习] 跳过（缺少依赖）: {e}\n\n')
except Exception as e:
    print(f'  方案C 运行失败: {e}')
    import traceback; traceback.print_exc()
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案C 注意力图结构学习] 运行失败: {e}\n\n')
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
