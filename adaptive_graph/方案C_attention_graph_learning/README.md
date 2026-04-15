# 方案C：注意力学习的图结构优化

## 概述

方案C基于 **Pro-GNN** (arXiv:2005.10203) 的可学习邻接矩阵思想和 **ASTGCN** (AAAI 2019) 的空间注意力机制，用可学习的注意力机制替代固定的 Bresenham 连边，让模型自动学习最优图结构。

## 核心思想

```
传统方法 (v2):  固定 Bresenham 连边 -> 固定图结构 -> GNN 训练
方案C:          候选池 + MLP 选择 -> 可学习边权重 -> 端到端联合优化
```

### 与 v2 的关键区别

| 特性 | v2 (ntl_graph_accel_v2) | 方案C |
|------|------------------------|-------|
| 节点选择 | 固定36个，距离排序+区域配额 | 候选池64个，MLP选择top-K |
| 边构建 | Bresenham 3D射线硬连接 | 特征相似度注意力软连接 |
| 图结构 | 固定不变 | 可学习，端到端优化 |
| 正则化 | 无 | 稀疏正则化 + 平滑正则化 |

## 文件结构

```
方案C_attention_graph_learning/
├── __init__.py          # 包入口，导出所有公共接口
├── config_c.py          # 方案C配置（数据、图构建、可学习图结构、GNN模型）
├── learnable_graph.py   # 可学习图结构模块（PyTorch nn.Module）
├── graph_builder_c.py   # 图构建器（候选池 + MLP 节点选择）
├── model_c.py           # GNN训练模型（GATv2 + 可学习图结构）
└── README.md            # 本文档
```

## 模块说明

### 1. config_c.py - 配置模块

包含四个配置类：

- **DataConfigC**: 数据配置（与 v2 兼容）
- **GraphBuildConfigC**: 图构建配置（候选池大小、MLP参数、Numba开关）
- **LearnableGraphConfigC**: 可学习图结构配置（节点选择器、边构建器、正则化）
- **GNNModelConfigC**: GNN模型配置（GATv2参数、训练参数、联合训练策略）

### 2. learnable_graph.py - 可学习图结构模块

三个核心 PyTorch 模块：

#### LearnableNodeSelector (MLP 节点选择器)
- 输入：候选节点特征 `(dt, dh, dw, value)`
- 通过位置编码 + 特征融合 + MLP 计算重要性分数
- 输出：top-K 节点索引 + 注意力权重
- 支持温度参数控制选择分布的锐度

#### LearnableEdgeBuilder (可学习边构建器)
- 替代 Bresenham 硬连接
- 基于特征投影 + 余弦相似度 + 距离编码计算边注意力权重
- KNN 稀疏化 + 自环 + 对称化
- 支持稀疏正则化和平滑正则化

#### GraphStructureLearner (图结构学习器)
- 整合节点选择 + 边构建的端到端模块
- 输出：邻接矩阵、边索引、正则化损失
- 支持边权重阈值过滤

### 3. graph_builder_c.py - 图构建器

图构建阶段的实现：

- **候选池收集**: 使用 Numba JIT 加速，从时空立方体中收集候选节点
- **MLP 节点选择**: 可选使用 MLP 从候选池中选择 top-K 节点
- **初始边构建**: 基于距离的 KNN 边（训练阶段由 LearnableEdgeBuilder 优化）
- **SubGraphC**: 与 v2 SubGraph 兼容的数据结构，额外携带候选池信息

### 4. model_c.py - GNN 训练模型

完整的训练/推理接口：

- **GATv2Encoder**: 多层 GATv2 图注意力编码器（支持残差连接、归一化）
- **ReadoutHead**: 图级读出层（注意力加权池化 + MLP）
- **AttentionGraphModel**: 完整模型（图结构学习 + GATv2 + 读出）
- **GraphDatasetC**: PyTorch Dataset（兼容 v2 SubGraph 格式）
- **Trainer**: 训练器（联合训练、早停、学习率调度、检查点管理）

## 使用方法

### 图构建阶段

```python
import numpy as np
from 方案C_attention_graph_learning import ConfigC, GraphBuilderC

# 配置
config = ConfigC()
config.input_path = "path/to/data.npy"
config.graph_build.candidate_pool_size = 64
config.graph_build.num_nodes = 36

# 加载数据
data = np.load(config.input_path).astype(np.float32)
valid_mask = ~np.isnan(data)

# 构建图
builder = GraphBuilderC(config, data, valid_mask)
positions = ...  # 目标位置数组 (N, 3)
subgraphs = builder.build_batch(positions)
```

### 训练阶段

```python
from 方案C_attention_graph_learning import (
    ConfigC, AttentionGraphModel, Trainer, GraphDatasetC
)

# 配置
config = ConfigC()
config.gnn_model.num_epochs = 200
config.gnn_model.batch_size = 64
config.gnn_model.learning_rate = 1e-3
config.gnn_model.joint_training = True

# 创建模型
model = AttentionGraphModel(config)

# 创建数据集
train_dataset = GraphDatasetC(train_subgraphs, max_candidates=64)
val_dataset = GraphDatasetC(val_subgraphs, max_candidates=64)

# 训练
trainer = Trainer(model, config)
history = trainer.fit(train_dataset, val_dataset)

# 推理
predictions = trainer.predict(test_loader)
```

### 与 v2 数据兼容

方案C 可以直接加载 v2 构建的 SubGraph 数据（无候选池信息时自动回退）：

```python
from 方案C_attention_graph_learning import GraphDatasetC, SubGraphC

# 从 v2 SubGraph 转换
subgraph_c = SubGraphC.from_v2_subgraph(v2_subgraph)

# 或直接传入 v2 SubGraph 列表（自动兼容）
dataset = GraphDatasetC(v2_subgraphs)
```

## 联合训练策略

方案C 支持图结构学习和 GNN 的端到端联合优化：

1. **联合训练** (`joint_training=True`): 图结构学习器和 GNN 同时优化
   - 图结构学习器使用更低的学习率（`graph_lr_ratio=0.1`）
   - 正则化损失（稀疏 + 平滑）防止过拟合
2. **分阶段训练** (`freeze_graph_after_epoch=100`): 前100轮联合训练，之后冻结图结构
3. **预热**: 前5轮使用 warmup 策略

## 正则化

- **稀疏正则化**: 邻接矩阵的 L1 范数，鼓励稀疏连接
- **平滑正则化**: 图拉普拉斯正则化，鼓励相邻节点特征相似
- **熵正则化**: 节点选择注意力分布的熵，鼓励集中选择

## 依赖

- Python >= 3.8
- PyTorch >= 1.10
- NumPy >= 1.20
- Numba >= 0.55 (可选，用于图构建加速)

## 参考文献

1. Pro-GNN: *Parameterized Graph Convolutional Networks for Graph Representation Learning*, arXiv:2005.10203
2. ASTGCN: *Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting*, AAAI 2019
