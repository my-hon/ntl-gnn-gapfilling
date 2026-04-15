#!/bin/bash
# ================================================================
# 脚本5: 测试自适应图结构方案 A/B/C
# 在 Featurize 服务器上运行（需先完成脚本1和脚本2）
# ================================================================
set -e

eval "$(conda shell.bash hook)"
conda activate ntl

WORKDIR="/home/featurize/app/ntl-work"
PROJECT_DIR="$WORKDIR/ntl-gnn-gapfilling"
DATA_DIR="$WORKDIR/data"
OUTPUT_BASE="$WORKDIR/run_results_adaptive"
mkdir -p "$OUTPUT_BASE"

NTL_DATA="$DATA_DIR/beijing_2020_ntl.npy"
QUALITY_DATA="$DATA_DIR/beijing_2020_quality.npy"

if [ ! -f "$NTL_DATA" ]; then
    echo "错误: 未找到数据文件 $NTL_DATA"
    exit 1
fi

echo "========================================"
echo "  自适应图结构方案测试: A / B / C"
echo "========================================"
echo "数据: $NTL_DATA"
echo "质量: $QUALITY_DATA"
echo "输出: $OUTPUT_BASE"
echo ""

RESULTS_FILE="$OUTPUT_BASE/adaptive_results.txt"
echo "自适应图结构方案测试" > "$RESULTS_FILE"
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

# ---- 通用数据加载与预处理 ----
python -c "
import sys, numpy as np
sys.path.insert(0, '$PROJECT_DIR')

# 加载数据
raw = np.load('$NTL_DATA')
quality = np.load('$QUALITY_DATA')

# 预处理: quality > 1 -> NaN, /10.0, /100.0
data = raw.copy().astype(np.float32)
data[quality > 1] = np.nan
data = data / 10.0 / 100.0

valid_mask = ~np.isnan(data)
print(f'数据形状: {data.shape}')
print(f'有效像素: {valid_mask.sum()}/{data.size} ({100*valid_mask.sum()/data.size:.1f}%)')
valid_values = data[valid_mask]
print(f'有效值范围: [{valid_values.min():.4f}, {valid_values.max():.4f}]')

# 采样 1000 个缺失位置（有效区域内的 NaN）
T, H, W = data.shape
et, eh, ew = 50, 50, 50  # 缓冲区
region = data[et:T-et, eh:H-eh, ew:W-ew]
missing_positions = np.argwhere(np.isnan(region))
missing_positions[:, 0] += et
missing_positions[:, 1] += eh
missing_positions[:, 2] += ew

rng = np.random.RandomState(0)
n_test = min(1000, len(missing_positions))
test_indices = rng.choice(len(missing_positions), size=n_test, replace=False)
test_pos = missing_positions[test_indices]

np.save('$OUTPUT_BASE/test_positions.npy', test_pos)
np.save('$OUTPUT_BASE/data_preprocessed.npy', data)
np.save('$OUTPUT_BASE/valid_mask.npy', valid_mask)
print(f'测试位置: {n_test} 个（从 {len(missing_positions)} 个缺失位置中采样）')
print('数据准备完成!')
"

echo ""
echo "========================================"
echo "  [1/3] 方案A - 质量驱动自适应节点数"
echo "========================================"

A_OUTPUT="$OUTPUT_BASE/scheme_a_output"
mkdir -p "$A_OUTPUT"

python -c "
import sys, time, os, numpy as np
sys.path.insert(0, '$PROJECT_DIR')

try:
    from adaptive_graph.scheme_a_quality_adaptive_nodes.config_a import ConfigA
    from adaptive_graph.scheme_a_quality_adaptive_nodes.graph_builder_a import GraphBuilderA

    config = ConfigA()
    config.input_path = '$NTL_DATA'
    config.output_dir = '$A_OUTPUT'
    config.cache_dir = '$OUTPUT_BASE/scheme_a_cache'

    data = np.load('$OUTPUT_BASE/data_preprocessed.npy')
    valid_mask = np.load('$OUTPUT_BASE/valid_mask.npy').astype(bool)
    test_pos = np.load('$OUTPUT_BASE/test_positions.npy')

    builder = GraphBuilderA(config, data, valid_mask)

    # 预计算质量图
    print('  预计算质量图...')
    t_quality = time.time()
    builder.precompute_quality()
    quality_time = time.time() - t_quality
    print(f'  质量图计算耗时: {quality_time:.2f}s')

    # 构建子图
    print(f'  构建 {len(test_pos)} 个子图...')
    t0 = time.time()
    graphs = []
    node_counts = []
    for pos in test_pos:
        g = builder.build_single(int(pos[0]), int(pos[1]), int(pos[2]))
        if g is not None:
            graphs.append(g)
            node_counts.append(g.num_nodes)
    elapsed = time.time() - t0

    speed = len(graphs) / max(elapsed, 0.001)
    print(f'  方案A 结果: {len(graphs)} 图, 耗时 {elapsed:.2f}s, 速度 {speed:.0f} 图/秒')
    if node_counts:
        print(f'  节点数: min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}, std={np.std(node_counts):.1f}')

    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案A 质量自适应节点]\n')
        f.write(f'  质量图计算: {quality_time:.2f}s\n')
        f.write(f'  构建图数: {len(graphs)}\n')
        f.write(f'  耗时: {elapsed:.2f}s\n')
        f.write(f'  速度: {speed:.0f} 图/秒\n')
        if node_counts:
            f.write(f'  节点数: min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}\n\n')

except Exception as e:
    print(f'  方案A 运行失败: {e}')
    import traceback; traceback.print_exc()
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案A 质量自适应节点] 运行失败: {e}\n\n')
"

echo ""
echo "========================================"
echo "  [2/3] 方案B - 空间异质性感知动态连边"
echo "========================================"

B_OUTPUT="$OUTPUT_BASE/scheme_b_output"
mkdir -p "$B_OUTPUT"

python -c "
import sys, time, os, numpy as np
sys.path.insert(0, '$PROJECT_DIR')

try:
    from adaptive_graph.scheme_b_dynamic_edge_construction.config_b import ConfigB
    from adaptive_graph.scheme_b_dynamic_edge_construction.graph_builder_b import GraphBuilderB

    config = ConfigB()
    config.input_path = '$NTL_DATA'
    config.output_dir = '$B_OUTPUT'
    config.cache_dir = '$OUTPUT_BASE/scheme_b_cache'

    data = np.load('$OUTPUT_BASE/data_preprocessed.npy')
    valid_mask = np.load('$OUTPUT_BASE/valid_mask.npy').astype(bool)
    test_pos = np.load('$OUTPUT_BASE/test_positions.npy')

    builder = GraphBuilderB(config, data, valid_mask)

    # 构建子图
    print(f'  构建 {len(test_pos)} 个子图...')
    t0 = time.time()
    graphs = []
    node_counts = []
    edge_counts = []
    het_indices = []
    for pos in test_pos:
        g = builder.build_single(int(pos[0]), int(pos[1]), int(pos[2]))
        if g is not None:
            graphs.append(g)
            node_counts.append(g.num_nodes)
            if hasattr(g, 'edge_index_src') and g.edge_index_src is not None:
                edge_counts.append(len(g.edge_index_src))
            if hasattr(g, 'heterogeneity_index'):
                het_indices.append(g.heterogeneity_index)
    elapsed = time.time() - t0

    speed = len(graphs) / max(elapsed, 0.001)
    print(f'  方案B 结果: {len(graphs)} 图, 耗时 {elapsed:.2f}s, 速度 {speed:.0f} 图/秒')
    if node_counts:
        print(f'  节点数: mean={np.mean(node_counts):.1f}')
    if edge_counts:
        print(f'  边数: mean={np.mean(edge_counts):.1f}')
    if het_indices:
        valid_het = [h for h in het_indices if h >= 0]
        if valid_het:
            print(f'  异质性指数: mean={np.mean(valid_het):.4f}, std={np.std(valid_het):.4f}')

    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案B 异质性动态连边]\n')
        f.write(f'  构建图数: {len(graphs)}\n')
        f.write(f'  耗时: {elapsed:.2f}s\n')
        f.write(f'  速度: {speed:.0f} 图/秒\n')
        if node_counts:
            f.write(f'  节点数: mean={np.mean(node_counts):.1f}\n')
        if edge_counts:
            f.write(f'  边数: mean={np.mean(edge_counts):.1f}\n')
        if het_indices:
            valid_het = [h for h in het_indices if h >= 0]
            if valid_het:
                f.write(f'  异质性指数: mean={np.mean(valid_het):.4f}\n\n')
            else:
                f.write('\n')
        else:
            f.write('\n')

except Exception as e:
    print(f'  方案B 运行失败: {e}')
    import traceback; traceback.print_exc()
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案B 异质性动态连边] 运行失败: {e}\n\n')
"

echo ""
echo "========================================"
echo "  [3/3] 方案C - 注意力图结构学习"
echo "========================================"

C_OUTPUT="$OUTPUT_BASE/scheme_c_output"
mkdir -p "$C_OUTPUT"

python -c "
import sys, time, os, numpy as np
sys.path.insert(0, '$PROJECT_DIR')

try:
    import torch
    from adaptive_graph.scheme_c_attention_graph_learning.config_c import ConfigC
    from adaptive_graph.scheme_c_attention_graph_learning.graph_builder_c import GraphBuilderC

    config = ConfigC()
    config.data.data_shape = (366, 560, 666)
    config.input_path = '$NTL_DATA'
    config.output_dir = '$C_OUTPUT'
    config.cache_dir = '$OUTPUT_BASE/scheme_c_cache'

    device = config.get_device()
    print(f'  使用设备: {device}')

    data = np.load('$OUTPUT_BASE/data_preprocessed.npy')
    valid_mask = np.load('$OUTPUT_BASE/valid_mask.npy').astype(bool)
    test_pos = np.load('$OUTPUT_BASE/test_positions.npy')

    builder = GraphBuilderC(config, data, valid_mask)

    # 构建子图（方案C较慢，取500个）
    n_test_c = min(500, len(test_pos))
    print(f'  构建 {n_test_c} 个子图...')
    t0 = time.time()
    graphs = []
    node_counts = []
    for pos in test_pos[:n_test_c]:
        g = builder.build_single(int(pos[0]), int(pos[1]), int(pos[2]))
        if g is not None:
            graphs.append(g)
            node_counts.append(g.num_nodes)
    elapsed = time.time() - t0

    speed = len(graphs) / max(elapsed, 0.001)
    print(f'  方案C 结果: {len(graphs)} 图, 耗时 {elapsed:.2f}s, 速度 {speed:.0f} 图/秒')
    if node_counts:
        print(f'  节点数: mean={np.mean(node_counts):.1f}')

    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'[方案C 注意力图结构学习]\n')
        f.write(f'  构建图数: {len(graphs)} (测试 {n_test_c} 个位置)\n')
        f.write(f'  耗时: {elapsed:.2f}s\n')
        f.write(f'  速度: {speed:.0f} 图/秒\n')
        if node_counts:
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
echo "  自适应方案测试汇总"
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
