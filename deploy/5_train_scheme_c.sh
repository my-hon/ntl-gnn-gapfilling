#!/bin/bash
# ================================================================
# 脚本5: 方案C 完整训练流程（支持图结构缓存）
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
GRAPH_CACHE_DIR="$OUTPUT_BASE/graphs_cache"
mkdir -p "$OUTPUT_BASE" "$GRAPH_CACHE_DIR"

NTL_DATA="$DATA_DIR/beijing_2020_ntl.npy"
QUALITY_DATA="$DATA_DIR/beijing_2020_quality.npy"

if [ ! -f "$NTL_DATA" ]; then
    echo "错误: 未找到数据文件 $NTL_DATA"
    exit 1
fi

# 解析参数
SKIP_BUILD=false
BUILD_ONLY=false
for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=true ;;
        --build-only) BUILD_ONLY=true ;;
    esac
done

echo "========================================"
echo "  方案C 完整训练流程"
echo "========================================"
echo "数据: $NTL_DATA"
echo "质量: $QUALITY_DATA"
echo "输出: $OUTPUT_BASE"
echo "图缓存: $GRAPH_CACHE_DIR"
echo "SKIP_BUILD=$SKIP_BUILD, BUILD_ONLY=$BUILD_ONLY"
echo ""

# 拉取最新代码
cd "$PROJECT_DIR"
git pull

python -c "
import sys, os, time, logging, pickle
import numpy as np
import torch

sys.path.insert(0, '$PROJECT_DIR')

# ---- 配置日志 ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger('scheme_c_train')

SKIP_BUILD = $SKIP_BUILD == 'true'
BUILD_ONLY = $BUILD_ONLY == 'true'
GRAPH_CACHE_DIR = '$GRAPH_CACHE_DIR'
OUTPUT_BASE = '$OUTPUT_BASE'

# ================================================================
# 阶段 1: 数据准备
# ================================================================
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

# ================================================================
# 阶段 2: 自然间断法采样
# ================================================================
logger.info('[2/5] 自然间断法采样训练数据...')

natural_breaks = [
    -float('inf'), 0.001, 0.00325, 0.0065, 0.0125,
    0.025, 0.1, float('inf')
]
sample_per_class = 20000
et, eh, ew = 50, 50, 50

region_data = data[et:T-et, eh:H-eh, ew:W-ew]
region_valid = valid_mask[et:T-et, eh:H-eh, ew:W-ew]

all_positions = []
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
    positions[:, 0] += et
    positions[:, 1] += eh
    positions[:, 2] += ew
    all_positions.append(positions)
    logger.info(f'  类别 {cls_idx+1}: 采样 {len(positions)} 个')

train_positions = np.vstack(all_positions)
logger.info(f'总训练样本: {len(train_positions)}')

# 划分训练/验证集 (8:2)
rng = np.random.RandomState(42)
indices = rng.permutation(len(train_positions))
split = int(len(train_positions) * 0.8)
train_idx, val_idx = indices[:split], indices[split:]
train_pos = train_positions[train_idx]
val_pos = train_positions[val_idx]
logger.info(f'训练集: {len(train_pos)}, 验证集: {len(val_pos)}')

# 保存采样位置（用于复现）
np.save(os.path.join(GRAPH_CACHE_DIR, 'train_positions.npy'), train_pos)
np.save(os.path.join(GRAPH_CACHE_DIR, 'val_positions.npy'), val_pos)

data_time = time.time() - t0
logger.info(f'数据准备耗时: {data_time:.1f}s')

# ================================================================
# 阶段 3: 构建图结构（支持缓存）
# ================================================================
TRAIN_GRAPH_PATH = os.path.join(GRAPH_CACHE_DIR, 'train_graphs.pkl')
VAL_GRAPH_PATH = os.path.join(GRAPH_CACHE_DIR, 'val_graphs.pkl')

train_graphs = None
val_graphs = None

if SKIP_BUILD and os.path.exists(TRAIN_GRAPH_PATH) and os.path.exists(VAL_GRAPH_PATH):
    logger.info('[3/5] 从缓存加载图结构...')
    t_load = time.time()
    with open(TRAIN_GRAPH_PATH, 'rb') as f:
        train_graphs = pickle.load(f)
    with open(VAL_GRAPH_PATH, 'rb') as f:
        val_graphs = pickle.load(f)
    logger.info(f'  训练图: {len(train_graphs)}, 验证图: {len(val_graphs)}, 加载耗时: {time.time()-t_load:.1f}s')
else:
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
    config.output_dir = OUTPUT_BASE
    config.model_save_dir = os.path.join(OUTPUT_BASE, 'checkpoints')
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

    # 保存到缓存
    logger.info('保存图结构到缓存...')
    t_save = time.time()
    with open(TRAIN_GRAPH_PATH, 'wb') as f:
        pickle.dump(train_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(VAL_GRAPH_PATH, 'wb') as f:
        pickle.dump(val_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    train_size = os.path.getsize(TRAIN_GRAPH_PATH) / 1024**3
    val_size = os.path.getsize(VAL_GRAPH_PATH) / 1024**3
    logger.info(f'  训练图: {train_size:.2f} GB, 验证图: {val_size:.2f} GB, 保存耗时: {time.time()-t_save:.1f}s')

if BUILD_ONLY:
    logger.info('--build-only 模式，跳过训练')
    logger.info('=' * 60)
    logger.info('图结构生成完成!')
    logger.info(f'训练图: {GRAPH_CACHE_DIR}/train_graphs.pkl')
    logger.info(f'验证图: {GRAPH_CACHE_DIR}/val_graphs.pkl')
    logger.info('=' * 60)
    sys.exit(0)

# ================================================================
# 阶段 4: 创建数据集
# ================================================================
logger.info('[4/5] 创建数据集...')

from adaptive_graph.方案C_attention_graph_learning.model_c import (
    GraphDatasetC, Trainer, AttentionGraphModel
)

# 重新创建 config（确保参数一致）
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
config.output_dir = OUTPUT_BASE
config.model_save_dir = os.path.join(OUTPUT_BASE, 'checkpoints')
config.seed = 42

train_dataset = GraphDatasetC(train_graphs, max_candidates=config.graph_build.candidate_pool_size)
val_dataset = GraphDatasetC(val_graphs, max_candidates=config.graph_build.candidate_pool_size)
logger.info(f'训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}')

# ================================================================
# 阶段 5: 训练模型
# ================================================================
logger.info('[5/5] 训练模型...')

model = AttentionGraphModel(config).to(device)
trainer = Trainer(model, config)

history = trainer.fit(train_dataset, val_dataset)

# ================================================================
# 保存结果
# ================================================================
logger.info('保存训练结果...')

history_path = os.path.join(OUTPUT_BASE, 'training_history.npy')
np.save(history_path, history)
logger.info(f'训练历史已保存: {history_path}')

best_loss = min(history['val_loss'])
best_mae = history['val_mae'][np.argmin(history['val_loss'])]
final_loss = history['train_loss'][-1]
final_mae = history['train_mae'][-1]

summary_path = os.path.join(OUTPUT_BASE, 'training_summary.txt')
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
