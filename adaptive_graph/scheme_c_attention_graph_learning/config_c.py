"""
方案C：注意力学习的图结构优化 - 配置模块
===========================================
灵感来源：Pro-GNN (arXiv:2005.10203) + ASTGCN (AAAI 2019)
核心思想：用可学习的注意力机制替代固定的Bresenham连边，让模型自动学习最优图结构。
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


# ============================================================
# 数据配置（与 v2 兼容）
# ============================================================

@dataclass
class DataConfigC:
    """数据相关配置（与 v2 DataConfig 兼容）"""
    data_shape: Tuple[int, int, int] = (366, 476, 520)
    buffer_size: int = 50
    temporal_buffer: int = 10
    nan_value: float = float('nan')
    feature_scale: float = 108.3
    edge_scale: float = 8.0


# ============================================================
# 图构建配置（方案C 特有）
# ============================================================

@dataclass
class GraphBuildConfigC:
    """
    方案C 图构建阶段配置。
    与 v2 的区别：
      - candidate_pool_size: 候选节点池大小（v2 固定36个，方案C生成64个候选）
      - num_nodes: 最终选择的节点数（从候选池中选 top-K）
      - use_mlp_selector: 是否使用 MLP 节点选择器（图构建阶段）
    """
    # 候选池参数
    candidate_pool_size: int = 64       # 候选节点池大小
    num_nodes: int = 36                 # 最终选择的节点数（top-K）
    num_regions: int = 6                # 空间区域数量
    initial_radius: int = 4             # 初始搜索半径
    max_radius: int = 20                # 最大搜索半径

    # 节点选择参数
    use_mlp_selector: bool = True       # 是否使用 MLP 节点选择器
    mlp_hidden_dim: int = 32            # MLP 隐藏层维度
    mlp_num_layers: int = 2             # MLP 层数（不含输入/输出层）

    # 边构建参数
    edge_knn: int = 4                   # KNN 边构建的 K 值
    edge_similarity_threshold: float = 0.1  # 特征相似度阈值

    # Numba 加速
    use_numba: bool = True              # 是否启用 Numba JIT


# ============================================================
# 可学习图结构配置（训练阶段）
# ============================================================

@dataclass
class LearnableGraphConfigC:
    """
    可学习图结构配置（PyTorch nn.Module 参数）。
    对应 learnable_graph.py 中的三个模块。
    """
    # ---- LearnableNodeSelector 参数 ----
    node_selector_input_dim: int = 4    # 输入特征维度 (dt, dh, dw, value)
    node_selector_hidden_dim: int = 64  # 隐藏层维度
    node_selector_output_dim: int = 16  # 输出维度（注意力分数投影）
    node_selector_num_layers: int = 2   # MLP 层数
    node_selector_top_k: int = 36       # 选择 top-K 个节点
    node_selector_temperature: float = 1.0  # softmax 温度参数

    # ---- LearnableEdgeBuilder 参数 ----
    edge_builder_feature_dim: int = 1   # 节点特征维度（标量值）
    edge_builder_hidden_dim: int = 32   # MLP 隐藏层维度
    edge_builder_knn_k: int = 4         # KNN 邻居数
    edge_builder_self_loop: bool = True # 是否添加自环
    edge_builder_symmetric: bool = True # 是否构建对称边
    edge_builder_sparsity: float = 0.3  # 稀疏度目标（用于正则化）

    # ---- GraphStructureLearner 参数 ----
    gsl_edge_threshold: float = 0.05    # 边权重阈值（低于此值的边被移除）
    gsl_sparsity_weight: float = 0.01   # 稀疏正则化权重
    gsl_smoothness_weight: float = 0.005  # 平滑正则化权重


# ============================================================
# GNN 模型配置（方案C）
# ============================================================

@dataclass
class GNNModelConfigC:
    """
    GATv2 + 可学习图结构的 GNN 模型配置。
    """
    # ---- GATv2 编码器 ----
    gat_hidden_dim: int = 64            # GATv2 隐藏层维度
    gat_num_heads: int = 4              # 注意力头数
    gat_num_layers: int = 2             # GATv2 层数
    gat_dropout: float = 0.1            # Dropout 比率
    gat_residual: bool = True           # 是否使用残差连接
    gat_norm: str = "layer"             # 归一化方式: "layer" / "batch" / None

    # ---- 读出层 ----
    readout_hidden_dim: int = 32        # 读出 MLP 隐藏层维度
    readout_num_layers: int = 2         # 读出 MLP 层数

    # ---- 输出 ----
    output_dim: int = 1                 # 输出维度（回归任务）

    # ---- 训练 ----
    learning_rate: float = 1e-3         # 学习率
    weight_decay: float = 1e-5          # 权重衰减
    scheduler_type: str = "cosine"      # 学习率调度器: "cosine" / "step" / None
    scheduler_step_size: int = 50       # StepLR 步长
    scheduler_gamma: float = 0.5        # StepLR 衰减因子
    warmup_epochs: int = 5              # 预热轮数

    # ---- 训练流程 ----
    batch_size: int = 64                # 批大小
    num_epochs: int = 200               # 训练轮数
    early_stopping_patience: int = 20   # 早停耐心值
    gradient_clip: float = 1.0          # 梯度裁剪

    # ---- 联合训练 ----
    joint_training: bool = True         # 是否联合训练图结构 + GNN
    graph_lr_ratio: float = 0.1         # 图结构学习率与 GNN 学习率的比值
    freeze_graph_after_epoch: int = 100 # 在此轮数后冻结图结构（-1 表示不冻结）


# ============================================================
# 总配置
# ============================================================

@dataclass
class ConfigC:
    """方案C 总配置"""
    data: DataConfigC = field(default_factory=DataConfigC)
    graph_build: GraphBuildConfigC = field(default_factory=GraphBuildConfigC)
    learnable_graph: LearnableGraphConfigC = field(default_factory=LearnableGraphConfigC)
    gnn_model: GNNModelConfigC = field(default_factory=GNNModelConfigC)

    # 路径配置
    input_path: str = ""
    output_dir: str = "./output_graphs_c"
    cache_dir: str = "./graph_cache_c"
    model_save_dir: str = "./model_checkpoints_c"
    seed: int = 42

    # 设备配置
    device: str = "auto"               # "auto" / "cuda" / "cpu"
    num_workers: int = 4               # 数据加载工作线程数

    def __post_init__(self):
        """自动创建必要的输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)

    def get_effective_temporal_range(self):
        """获取有效时间范围"""
        T = self.data.data_shape[0]
        buf = min(self.data.temporal_buffer, T // 2)
        return buf, T - buf

    def get_effective_spatial_range(self):
        """获取有效空间范围"""
        buf = min(self.data.buffer_size,
                  min(self.data.data_shape[1], self.data.data_shape[2]) // 2)
        H, W = self.data.data_shape[1], self.data.data_shape[2]
        return buf, H - buf, buf, W - buf

    def get_device(self):
        """获取计算设备"""
        if self.device == "auto":
            import torch
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import torch
        return torch.device(self.device)
