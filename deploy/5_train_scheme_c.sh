#!/bin/bash
# ================================================================
# 脚本5: 方案C 完整训练流程
# 在 Featurize 服务器上运行（需先完成脚本1和脚本2）
# ================================================================
set -e

eval "$(conda shell.bash hook)"
conda activate ntl

WORKDIR="/home/featurize/app/ntl-work"
PROJECT_DIR="$WORKDIR/ntl-gnn-gapfilling"
DATA_DIR="$WORKDIR/data"
OUTPUT_BASE="$WORKDIR/scheme_c_train"
mkdir -p "$OUTPUT_BASE"

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

# 拉取最新代码
cd "$PROJECT_DIR"
git pull

python -c "
import sys, os, time, logging
import numpy as np
import torch

sys.path.insert(0, '$PROJECT_DIR')

# ---- 配置日志 ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('scheme_c_train')

# ---- 1. 加载并预处理数据 ----
logger.info('=' * 60)
logger.info('方案C 完整训练流程')
logger.info('=' * 60)

logger.info('[1/5] 加载并预处理数据...')
t0 = time.time()
raw = np.load('$NTL_DATA')
quality = np.load('$QUALITY_DATA')

data = raw.copy().astype(np.float32)
data[quality > 1] = np.nan
data = data / 10.0 / 100.0

valid_mask = ~np.isnan(data)
T, H, W = data.shape
logger.info(f'数据形状: {data.shape}, 有效像素: {valid_mask.sum()}/{data.size} ({100*valid_mask.sum()/data.size:.1f}%)')

# ---- 2. 自然间断法采样训练数据 ----
logger.info('[2/5] 自然间断法采样训练数据...')

natural_breaks = [
    -float('inf'), 0.001, 0.00325, 0.0065, 0.0125,
    0.025, 0.1, float('inf')
]
sample_per_class = 20000
et, eh, ew = 50, 50, 50  # 缓冲区

# 有效区域
region_data = data[et:T-et, eh:H-eh, ew:W-ew]
region_valid = valid_mask[et:T-et, eh:H-eh, ew:W-ew]

all_positions = []
all_values = []

for cls_idx in range(len(natural_breaks) - 1):
    lo, hi = natural_breaks[cls_idx], natural_breaks[cls_idx + 1]
    mask = region_valid & (region_data >= lo) & (region_data < hi)
    positions = np.argwhere(mask)

    if len(positions) == 0:
        logger.info(f'  类别 {cls_idx+1}: 无有效像素')
        continue

    if len(positions) > sample_per_class:
        rng = np.random.RandomState(cls_idx)
        indices = rng.choice(len(positions), size=sample_per_class, replace=False)
        positions = positions[indices]

    # 加回缓冲区偏移
    positions[:, 0] += et
    positions[:, 1] += eh
    positions[:, 2] += ew

    all_positions.append(positions)
    all_values.extend(data[positions[:, 0], positions[:, 1], positions[:, 2]].tolist())
    logger.info(f'  类别 {cls_idx+1}: 采样 {len(positions)} 个')

train_positions = np.vstack(all_positions)
train_values = np.array(all_values, dtype=np.float32)
logger.info(f'总训练样本: {len(train_positions)}')

# 划分训练/验证集 (8:2)
rng = np.random.RandomState(42)
indices = rng.permutation(len(train_positions))
split = int(len(train_positions) * 0.8)
train_idx, val_idx = indices[:split], indices[split:]

train_pos = train_positions[train_idx]
val_pos = train_positions[val_idx]
logger.info(f'训练集: {len(train_pos)}, 验证集: {len(val_pos)}')

data_time = time.time() - t0
logger.info(f'数据准备耗时: {data_time:.1f}s')

# ---- 3. 构建训练子图 ----
logger.info('[3/5] 构建训练子图...')

from adaptive_graph.方案C_attention_graph_learning.config_c import ConfigC
from adaptive_graph.方案C_attention_graph_learning.graph_builder_c import GraphBuilderC

config = ConfigC()
config.data.data_shape = (T, H, W)
config.graph_build.candidate_pool_size = 64
config.graph_build.num_nodes = 36
config.graph_build.use_mlp_selector = True
config.graph_build.mlp_hidden_dim = 32
config.graph_build.mlp_num_layers = 2
config.gnn_model.num_epochs = 200
config.gnn_model.batch_size = 64
config.gnn_model.learning_rate = 1e-3
config.gnn_model.early_stopping_patience = 20
config.gnn_model.gat_hidden_dim = 64
config.gnn_model.gat_num_heads = 4
config.gnn_model.gat_num_layers = 2
config.output_dir = '$OUTPUT_BASE'
config.model_save_dir = '$OUTPUT_BASE/checkpoints'
config.seed = 42

device = config.get_device()
logger.info(f'使用设备: {device}')

builder = GraphBuilderC(config, data, valid_mask)

def build_graphs(positions, desc='构建'):
    graphs = []
    total = len(positions)
    t_start = time.time()
    for i, pos in enumerate(positions):
        g = builder.build_single(int(pos[0]), int(pos[1]), int(pos[2]))
        if g is not None:
            graphs.append(g)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t_start
            speed = (i + 1) / max(elapsed, 0.001)
            logger.info(f'  {desc}: {i+1}/{total} ({speed:.0f} 图/秒), 已构建: {len(graphs)}')
    elapsed = time.time() - t_start
    speed = len(graphs) / max(elapsed, 0.001)
    logger.info(f'  {desc} 完成: {len(graphs)} 图, 耗时 {elapsed:.1f}s, 速度 {speed:.0f} 图/秒')
    return graphs

logger.info('构建训练集子图...')
train_graphs = build_graphs(train_pos, '训练集')

logger.info('构建验证集子图...')
val_graphs = build_graphs(val_pos, '验证集')

# ---- 4. 创建数据集 ----
logger.info('[4/5] 创建数据集...')

from adaptive_graph.方案C_attention_graph_learning.model_c import (
    GraphDatasetC, Trainer, AttentionGraphModel
)

train_dataset = GraphDatasetC(train_graphs, max_candidates=config.graph_build.candidate_pool_size)
val_dataset = GraphDatasetC(val_graphs, max_candidates=config.graph_build.candidate_pool_size)
logger.info(f'训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}')

# ---- 5. 训练模型 ----
logger.info('[5/5] 训练模型...')

model = AttentionGraphModel(config).to(device)
trainer = Trainer(model, config)

history = trainer.fit(train_dataset, val_dataset)

# ---- 保存结果 ----
logger.info('保存训练结果...')

# 保存训练历史
history_path = os.path.join('$OUTPUT_BASE', 'training_history.npy')
np.save(history_path, history)
logger.info(f'训练历史已保存: {history_path}')

# 保存最终指标
best_loss = min(history['val_loss'])
best_mae = history['val_mae'][np.argmin(history['val_loss'])]
final_loss = history['train_loss'][-1]
final_mae = history['train_mae'][-1]

summary_path = os.path.join('$OUTPUT_BASE', 'training_summary.txt')
with open(summary_path, 'w') as f:
    f.write('方案C 训练结果摘要\n')
    f.write('=' * 40 + '\n\n')
    f.write(f'数据: $NTL_DATA\n')
    f.write(f'训练样本: {len(train_graphs)}\n')
    f.write(f'验证样本: {len(val_graphs)}\n')
    f.write(f'设备: {device}\n\n')
    f.write(f'最佳验证 Loss: {best_loss:.6f}\n')
    f.write(f'最佳验证 MAE:  {best_mae:.4f}\n')
    f.write(f'最终训练 Loss: {final_loss:.6f}\n')
    f.write(f'最终训练 MAE:  {final_mae:.4f}\n')
    f.write(f'训练轮数: {len(history[\"train_loss\"])}\n\n')
    f.write('逐轮结果:\n')
    f.write(f'{\"Epoch\":>6} {\"Train_Loss\":>12} {\"Val_Loss\":>12} {\"Train_MAE\":>10} {\"Val_MAE\":>10}\n')
    for i in range(len(history['train_loss'])):
        f.write(f'{i+1:>6} {history[\"train_loss\"][i]:>12.6f} {history[\"val_loss\"][i]:>12.6f} '
                f'{history[\"train_mae\"][i]:>10.4f} {history[\"val_mae\"][i]:>10.4f}\n')

logger.info(f'训练摘要已保存: {summary_path}')
logger.info('=' * 60)
logger.info(f'训练完成! 最佳验证 Loss: {best_loss:.6f}, MAE: {best_mae:.4f}')
logger.info('=' * 60)
"
