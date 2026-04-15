"""
方案C：注意力学习的图结构优化
================================
基于 PyTorch 的可学习图结构 + GATv2 GNN 模型。

灵感来源：
  - Pro-GNN (arXiv:2005.10203): 可学习邻接矩阵
  - ASTGCN (AAAI 2019): 空间注意力机制

核心思想：
  用可学习的注意力机制替代固定的 Bresenham 连边，
  让模型自动学习最优图结构。

模块说明：
  - config_c.py: 方案C 配置（数据、图构建、可学习图结构、GNN 模型）
  - learnable_graph.py: 可学习图结构模块（LearnableNodeSelector, LearnableEdgeBuilder, GraphStructureLearner）
  - graph_builder_c.py: 图构建器（候选池 + MLP 节点选择，Numba 加速）
  - model_c.py: GNN 训练模型（GATv2 + 可学习图结构，完整训练/推理接口）

使用示例：
  >>> from 方案C_attention_graph_learning import ConfigC, GraphBuilderC, AttentionGraphModel, Trainer, GraphDatasetC
  >>> config = ConfigC()
  >>> # 图构建阶段
  >>> builder = GraphBuilderC(config, data, valid_mask)
  >>> subgraphs = builder.build_batch(positions)
  >>> # 训练阶段
  >>> model = AttentionGraphModel(config)
  >>> trainer = Trainer(model, config)
  >>> dataset = GraphDatasetC(subgraphs)
  >>> history = trainer.fit(dataset, val_dataset)
"""

__version__ = "1.0.0"

from .config_c import (
    ConfigC,
    DataConfigC,
    GraphBuildConfigC,
    LearnableGraphConfigC,
    GNNModelConfigC,
)

from .learnable_graph import (
    LearnableNodeSelector,
    LearnableEdgeBuilder,
    GraphStructureLearner,
    build_mlp,
)

from .graph_builder_c import (
    GraphBuilderC,
    SubGraphC,
)

from .model_c import (
    GATv2Layer,
    GATv2Encoder,
    ReadoutHead,
    AttentionGraphModel,
    GraphDatasetC,
    Trainer,
    collate_fn_c,
)

__all__ = [
    # 配置
    "ConfigC",
    "DataConfigC",
    "GraphBuildConfigC",
    "LearnableGraphConfigC",
    "GNNModelConfigC",
    # 可学习图结构
    "LearnableNodeSelector",
    "LearnableEdgeBuilder",
    "GraphStructureLearner",
    "build_mlp",
    # 图构建器
    "GraphBuilderC",
    "SubGraphC",
    # 模型
    "GATv2Layer",
    "GATv2Encoder",
    "ReadoutHead",
    "AttentionGraphModel",
    "GraphDatasetC",
    "Trainer",
    "collate_fn_c",
]
