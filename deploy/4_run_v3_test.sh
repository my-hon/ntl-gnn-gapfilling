#!/bin/bash
# ================================================================
# 脚本4: 运行 v3 图构建（对齐参考算法）
# 在 Featurize 服务器上运行（需先完成脚本1和脚本2）
# ================================================================
set -e

eval "$(conda shell.bash hook)"
conda activate ntl

WORKDIR="/home/featurize/app/ntl-work"
PROJECT_DIR="$WORKDIR/ntl-gnn-gapfilling"
DATA_DIR="$WORKDIR/data"
OUTPUT_BASE="$WORKDIR/run_results_v3"
mkdir -p "$OUTPUT_BASE"

NTL_DATA="$DATA_DIR/beijing_2020_ntl.npy"
QUALITY_DATA="$DATA_DIR/beijing_2020_quality.npy"

if [ ! -f "$NTL_DATA" ]; then
    echo "错误: 未找到数据文件 $NTL_DATA"
    exit 1
fi

echo "========================================"
echo "  v3 图构建测试（对齐参考算法）"
echo "========================================"
echo "数据: $NTL_DATA"
echo "质量: $QUALITY_DATA"
echo ""

# ---- 拉取最新代码 ----
cd "$PROJECT_DIR"
git pull

# ---- 测试1: 数据预处理验证 ----
echo "--- [测试1] 数据预处理验证 ---"
python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
import numpy as np
from ntl_graph_accel_v3.data_loader import DataLoader

loader = DataLoader()
data = loader.load_from_npy('$NTL_DATA', '$QUALITY_DATA')
print(f'数据形状: {data.shape}')
print(f'有效像素: {np.count_nonzero(~np.isnan(data))}/{data.size} ({100*np.count_nonzero(~np.isnan(data))/data.size:.1f}%)')
valid = data[~np.isnan(data)]
print(f'有效值范围: [{valid.min():.4f}, {valid.max():.4f}]')
print(f'有效值均值: {valid.mean():.4f}')
print(f'有效值中位数: {np.median(valid):.4f}')
print('数据预处理验证通过!')
"

# ---- 测试2: 单图构建验证 ----
echo ""
echo "--- [测试2] 单图构建验证 ---"
python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
import numpy as np
from ntl_graph_accel_v3.data_loader import DataLoader
from ntl_graph_accel_v3.graph_builder import GraphBuilder, process_index
from ntl_graph_accel_v3.config import Config

config = Config()
config.data_shape = (366, 560, 666)

# 加载数据
loader = DataLoader()
data = loader.load_from_npy('$NTL_DATA', '$QUALITY_DATA')

# 选择一个有效位置
T, H, W = data.shape
tc, hc, wc = 100, 280, 333  # 中间位置

builder = GraphBuilder(config)
graph = builder.build_single(data, tc, hc, wc)

if graph is not None:
    print(f'节点数: {graph[\"node_features\"].shape[0]} (1中心 + {graph[\"node_features\"].shape[0]-1}邻居)')
    print(f'边数: {graph[\"edge_index\"].shape[1]}')
    print(f'中心特征: {graph[\"node_features\"][0]}')
    print(f'ground_truth[0]: {graph[\"ground_truth\"][0]}')
    print(f'位置: {graph[\"position\"]}')
    print(f'node_features shape: {graph[\"node_features\"].shape}')
    print(f'edge_index shape: {graph[\"edge_index\"].shape}')
    print(f'edge_attr shape: {graph[\"edge_attr\"].shape}')
    print('单图构建验证通过!')
else:
    print('单图构建失败!')
" 2>&1

# ---- 测试3: 小批量训练数据生成（100个/类） ----
echo ""
echo "--- [测试3] 小批量训练数据生成（100个/类） ---"
python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
import numpy as np
import time
from ntl_graph_accel_v3.data_loader import DataLoader
from ntl_graph_accel_v3.graph_builder import GraphBuilder
from ntl_graph_accel_v3.config import Config

config = Config()
config.data_shape = (366, 560, 666)
config.sample_per_class = 100  # 小批量测试
config.n_jobs = 4

loader = DataLoader()
data = loader.load_from_npy('$NTL_DATA', '$QUALITY_DATA')

# 自然间断法采样
print('自然间断法采样...')
positions_by_class = loader.natural_breaks_sample(data, config)
total = sum(len(p) for p in positions_by_class)
print(f'总采样数: {total}')

# 构建图
builder = GraphBuilder(config)
t0 = time.time()
for cls_idx, positions in enumerate(positions_by_class):
    cls_t0 = time.time()
    graphs = []
    for pos in positions:
        g = builder.build_single(data, int(pos[0]), int(pos[1]), int(pos[2]))
        if g is not None:
            graphs.append(g)
    cls_elapsed = time.time() - cls_t0
    speed = len(graphs) / max(cls_elapsed, 0.001)
    print(f'  类别 {cls_idx+1}: {len(graphs)} 图, {cls_elapsed:.2f}s, {speed:.0f} 图/秒')

total_elapsed = time.time() - t0
print(f'总耗时: {total_elapsed:.2f}s')
print('训练数据生成测试通过!')
" 2>&1

echo ""
echo "========================================"
echo "  v3 测试完成!"
echo "========================================"
