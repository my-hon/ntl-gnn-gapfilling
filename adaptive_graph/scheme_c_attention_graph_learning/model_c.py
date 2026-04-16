"""
方案C：GNN 训练模型（GATv2 + 可学习图结构）
=============================================
提供完整的训练/推理接口，实现图结构学习 + GNN 的端到端联合优化。

核心组件：
  1. GATv2Encoder: GATv2 图注意力编码器
  2. ReadoutHead: 图级读出层（回归任务）
  3. AttentionGraphModel: 完整模型（图结构学习 + GATv2 + 读出）
  4. GraphDataset / GraphDataLoader: 数据加载工具
  5. Trainer: 训练器（支持联合训练、早停、学习率调度）
"""

import os
import math
import time
import pickle
import logging
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .config_c import ConfigC, GNNModelConfigC
from .learnable_graph import GraphStructureLearner

logger = logging.getLogger(__name__)


# ============================================================
# 1. GATv2Encoder - GATv2 图注意力编码器
# ============================================================

class GATv2Layer(nn.Module):
    """
    单层 GATv2（GAT v2: Graph Attention Network v2）。

    与标准 GAT 的区别：
      - GATv2 使用动态注意力机制，先拼接再计算注意力
      - 公式: e_{ij} = LeakyReLU(a^T [W h_i || W h_j])
      - 标准 GAT: e_{ij} = LeakyReLU(a^T [W h_i || W h_j])（静态注意力）

    参数
    ----
    in_dim : int
        输入特征维度
    out_dim : int
        输出特征维度
    num_heads : int
        注意力头数
    dropout : float
        Dropout 比率
    residual : bool
        是否使用残差连接
    norm : str
        归一化方式
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
        norm: str = "layer",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim 必须能被 num_heads 整除"

        # 线性变换
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        # 注意力参数（GATv2: 先拼接再计算）
        self.att_src = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, num_heads, self.head_dim))

        # 注意力偏置
        self.att_bias = nn.Parameter(torch.empty(1, num_heads, 1))

        # LeakyReLU
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # 残差连接
        self.residual = residual
        if residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)

        # 归一化
        self.norm = None
        if norm == "layer":
            self.norm = nn.LayerNorm(out_dim)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.att_bias)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        x : torch.Tensor
            节点特征，形状 (batch_size, num_nodes, in_dim)
        adj : torch.Tensor
            邻接矩阵，形状 (batch_size, num_nodes, num_nodes)

        返回
        ----
        out : torch.Tensor
            输出特征，形状 (batch_size, num_nodes, out_dim)
        """
        batch_size, num_nodes, _ = x.shape

        # 线性变换
        h = self.W(x)  # (B, N, out_dim)
        h = h.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        h = h.permute(0, 2, 1, 3)  # (B, heads, N, head_dim)

        # GATv2 动态注意力: e_{ij} = a^T LeakyReLU(W h_i || W h_j)
        # 注意力分数 = att_src * h_i + att_dst * h_j + bias
        if h.shape[-1] != self.att_src.shape[-1]:
            raise ValueError(
                f"GATv2Layer dimension mismatch: h head_dim={h.shape[-1]}, "
                f"att_src dim={self.att_src.shape[-1]}, "
                f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
                f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
                f"x.shape={x.shape}, h.shape={h.shape}"
            )
        attn_src = (h * self.att_src).sum(dim=-1)  # (B, heads, N)
        attn_dst = (h * self.att_dst).sum(dim=-1)  # (B, heads, N)

        # 广播计算注意力分数
        # attn: (B, heads, N, N)
        attn = attn_src.unsqueeze(-1) + attn_dst.unsqueeze(-2) + self.att_bias
        attn = self.leaky_relu(attn)

        # 用邻接矩阵掩码
        # 扩展 adj 到多头: (B, 1, N, N)
        adj_mask = adj.unsqueeze(1)
        attn = attn.masked_fill(adj_mask == 0, float('-inf'))

        # Softmax 归一化
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 加权聚合
        # (B, heads, N, head_dim) * (B, heads, N, 1) -> sum
        out = h * attn.unsqueeze(-1)
        out = out.sum(dim=2)  # (B, heads, head_dim)
        out = out.permute(0, 2, 1)  # (B, head_dim, heads)
        out = out.reshape(batch_size, num_nodes, self.out_dim)
        out = self.out_dropout(out)

        # 残差连接
        if self.residual:
            res = x
            if hasattr(self, 'res_proj'):
                res = self.res_proj(res)
            out = out + res

        # 归一化
        if self.norm is not None:
            if isinstance(self.norm, nn.LayerNorm):
                out = self.norm(out)
            else:
                # BatchNorm1d 需要 (B*N, C)
                out = out.permute(0, 2, 1)  # (B, out_dim, N)
                out = self.norm(out)
                out = out.permute(0, 2, 1)  # (B, N, out_dim)

        return out


class GATv2Encoder(nn.Module):
    """
    多层 GATv2 编码器。

    参数
    ----
    input_dim : int
        输入特征维度
    hidden_dim : int
        隐藏层维度
    num_layers : int
        层数
    num_heads : int
        注意力头数
    dropout : float
        Dropout 比率
    residual : bool
        是否使用残差连接
    norm : str
        归一化方式
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
        norm: str = "layer",
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # 第一层: input_dim -> hidden_dim
        self.layers.append(GATv2Layer(
            in_dim=input_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            residual=(input_dim == hidden_dim) and residual,
            norm=norm,
        ))

        # 中间层: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(GATv2Layer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                residual=residual,
                norm=norm,
            ))

        # 最后一层: hidden_dim -> hidden_dim（无归一化）
        if num_layers > 1:
            self.layers.append(GATv2Layer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                residual=residual,
                norm=None,  # 最后一层不做归一化
            ))

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        x : torch.Tensor
            节点特征，形状 (batch_size, num_nodes, input_dim)
        adj : torch.Tensor
            邻接矩阵，形状 (batch_size, num_nodes, num_nodes)

        返回
        ----
        out : torch.Tensor
            节点嵌入，形状 (batch_size, num_nodes, hidden_dim)
        """
        h = x
        for layer in self.layers:
            h = layer(h, adj)
        return h


# ============================================================
# 2. ReadoutHead - 图级读出层
# ============================================================

class ReadoutHead(nn.Module):
    """
    图级读出层，将节点嵌入聚合为图级表示。

    支持多种读出策略：
      - mean: 平均池化
      - max: 最大池化
      - sum: 求和池化
      - attention: 注意力加权池化

    参数
    ----
    input_dim : int
        输入特征维度
    hidden_dim : int
        隐藏层维度
    output_dim : int
        输出维度
    num_layers : int
        MLP 层数
    readout_type : str
        读出策略
    dropout : float
        Dropout 比率
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 32,
        output_dim: int = 1,
        num_layers: int = 2,
        readout_type: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.readout_type = readout_type
        self.input_dim = input_dim

        # 注意力读出
        if readout_type == "attention":
            self.attention_weights = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        # 输出 MLP
        self.output_mlp = nn.Sequential()
        in_d = input_dim
        for i in range(num_layers):
            out_d = hidden_dim if i < num_layers - 1 else output_dim
            self.output_mlp.add_module(f"linear_{i}", nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                self.output_mlp.add_module(f"relu_{i}", nn.ReLU())
                self.output_mlp.add_module(f"dropout_{i}", nn.Dropout(dropout))
            in_d = out_d

    def forward(
        self,
        node_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        node_embeddings : torch.Tensor
            节点嵌入，形状 (batch_size, num_nodes, input_dim)
        mask : torch.Tensor, optional
            有效节点掩码，形状 (batch_size, num_nodes)

        返回
        ----
        output : torch.Tensor
            图级输出，形状 (batch_size, output_dim)
        """
        if self.readout_type == "attention":
            # 注意力加权池化
            attn_scores = self.attention_weights(node_embeddings).squeeze(-1)  # (B, N)
            if mask is not None:
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)  # (B, N)
            graph_repr = (node_embeddings * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, input_dim)
        elif self.readout_type == "mean":
            if mask is not None:
                node_embeddings = node_embeddings * mask.unsqueeze(-1).float()
                graph_repr = node_embeddings.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            else:
                graph_repr = node_embeddings.mean(dim=1)
        elif self.readout_type == "max":
            if mask is not None:
                node_embeddings = node_embeddings.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            graph_repr = node_embeddings.max(dim=1)[0]
        else:  # sum
            if mask is not None:
                node_embeddings = node_embeddings * mask.unsqueeze(-1).float()
            graph_repr = node_embeddings.sum(dim=1)

        # 输出 MLP
        output = self.output_mlp(graph_repr)
        return output


# ============================================================
# 3. AttentionGraphModel - 完整模型
# ============================================================

class AttentionGraphModel(nn.Module):
    """
    方案C 完整模型：图结构学习 + GATv2 + 读出。

    工作流程：
      1. GraphStructureLearner: 候选池 -> 选择节点 + 构建边
      2. GATv2Encoder: 节点特征 -> 节点嵌入
      3. ReadoutHead: 节点嵌入 -> 预测值

    参数
    ----
    config : ConfigC
        方案C 配置
    """

    def __init__(self, config: ConfigC):
        super().__init__()
        self.config = config
        gnn_cfg = config.gnn_model
        gsl_cfg = config.learnable_graph

        # ---- 图结构学习器 ----
        self.graph_learner = GraphStructureLearner(
            node_selector_cfg={
                "input_dim": gsl_cfg.node_selector_input_dim,
                "hidden_dim": gsl_cfg.node_selector_hidden_dim,
                "output_dim": gsl_cfg.node_selector_output_dim,
                "num_layers": gsl_cfg.node_selector_num_layers,
                "top_k": gsl_cfg.node_selector_top_k,
                "temperature": gsl_cfg.node_selector_temperature,
            },
            edge_builder_cfg={
                "feature_dim": gsl_cfg.edge_builder_feature_dim,
                "hidden_dim": gsl_cfg.edge_builder_hidden_dim,
                "knn_k": gsl_cfg.edge_builder_knn_k,
                "self_loop": gsl_cfg.edge_builder_self_loop,
                "symmetric": gsl_cfg.edge_builder_symmetric,
            },
            edge_threshold=gsl_cfg.gsl_edge_threshold,
            sparsity_weight=gsl_cfg.gsl_sparsity_weight,
            smoothness_weight=gsl_cfg.gsl_smoothness_weight,
        )

        # ---- GATv2 编码器 ----
        self.encoder = GATv2Encoder(
            input_dim=gnn_cfg.gat_hidden_dim,  # 使用 hidden_dim 作为输入（与图结构输出对齐）
            hidden_dim=gnn_cfg.gat_hidden_dim,
            num_layers=gnn_cfg.gat_num_layers,
            num_heads=gnn_cfg.gat_num_heads,
            dropout=gnn_cfg.gat_dropout,
            residual=gnn_cfg.gat_residual,
            norm=gnn_cfg.gat_norm,
        )

        # ---- 特征投影层（将标量特征投影到 GATv2 输入维度）----
        self.feature_projection = nn.Sequential(
            nn.Linear(1, gnn_cfg.gat_hidden_dim),
            nn.GELU(),
            nn.Linear(gnn_cfg.gat_hidden_dim, gnn_cfg.gat_hidden_dim),
        )

        # ---- 读出层 ----
        self.readout = ReadoutHead(
            input_dim=gnn_cfg.gat_hidden_dim,
            hidden_dim=gnn_cfg.readout_hidden_dim,
            output_dim=gnn_cfg.output_dim,
            num_layers=gnn_cfg.readout_num_layers,
            readout_type="attention",
            dropout=gnn_cfg.gat_dropout,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """参数初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    nn.init.xavier_uniform_(m.weight, gain=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_offsets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        参数
        ----
        candidate_features : torch.Tensor
            候选节点特征，形状 (batch_size, num_candidates, 4)
            最后一维为 (dt, dh, dw, value)
        candidate_offsets : torch.Tensor
            候选节点偏移量，形状 (batch_size, num_candidates, 3)
        mask : torch.Tensor, optional
            有效候选节点掩码，形状 (batch_size, num_candidates)

        返回
        ----
        result : dict
            - "prediction": 预测值 (B, 1)
            - "node_embeddings": 节点嵌入 (B, K, hidden_dim)
            - "adj_matrix": 邻接矩阵 (B, K, K)
            - "selected_indices": 选中的节点索引 (B, K)
            - "reg_loss": 图结构正则化损失
        """
        # ---- 1. 图结构学习 ----
        gsl_result = self.graph_learner(
            candidate_features, candidate_offsets, mask
        )

        # ---- 2. 获取邻接矩阵 ----
        adj, adj_normalized = self.graph_learner.get_adjacency_for_gnn(gsl_result)

        # ---- 3. 特征投影 ----
        selected_features = gsl_result["selected_features"]  # (B, K, 4)
        node_values = selected_features[:, :, -1:]  # (B, K, 1) - 只取 value
        projected_features = self.feature_projection(node_values)  # (B, K, hidden_dim)

        # ---- 4. GATv2 编码 ----
        node_embeddings = self.encoder(projected_features, adj_normalized)

        # ---- 5. 读出 ----
        prediction = self.readout(node_embeddings)

        return {
            "prediction": prediction,
            "node_embeddings": node_embeddings,
            "adj_matrix": adj,
            "adj_normalized": adj_normalized,
            "selected_indices": gsl_result["selected_indices"],
            "selected_scores": gsl_result["selected_scores"],
            "attention_weights": gsl_result["attention_weights"],
            "reg_loss": gsl_result["reg_loss"],
        }


# ============================================================
# 4. GraphDataset / GraphDataLoader - 数据加载工具
# ============================================================

class GraphDatasetC(Dataset):
    """
    方案C 图数据集。

    从方案C的 SubGraphC 列表中构建 PyTorch Dataset。
    每个样本包含候选节点池的特征和偏移。

    参数
    ----
    subgraphs : List[SubGraphC]
        子图列表
    max_candidates : int, optional
        最大候选数量（用于 padding）
    """

    def __init__(
        self,
        subgraphs: List,
        max_candidates: int = 64,
    ):
        self.subgraphs = subgraphs
        self.max_candidates = max_candidates

        # 统计信息
        self.num_nodes_list = [sg.num_nodes for sg in subgraphs]
        self.avg_nodes = np.mean(self.num_nodes_list) if self.num_nodes_list else 0

    def __len__(self):
        return len(self.subgraphs)

    def __getitem__(self, idx):
        sg = self.subgraphs[idx]

        # ---- 候选节点特征 ----
        if sg.candidate_features is not None and sg.candidate_offsets is not None:
            num_candidates = len(sg.candidate_features)
            # 构造输入: (dt, dh, dw, value)
            features = np.zeros((self.max_candidates, 4), dtype=np.float32)
            offsets = np.zeros((self.max_candidates, 3), dtype=np.float32)
            mask = np.zeros(self.max_candidates, dtype=np.bool_)

            features[:num_candidates, :3] = sg.candidate_offsets.astype(np.float32)
            features[:num_candidates, 3] = sg.candidate_features
            offsets[:num_candidates] = sg.candidate_offsets.astype(np.float32)
            mask[:num_candidates] = True
        else:
            # 兼容 v2 SubGraph（无候选池信息）
            num_nodes = sg.num_nodes
            features = np.zeros((self.max_candidates, 4), dtype=np.float32)
            offsets = np.zeros((self.max_candidates, 3), dtype=np.float32)
            mask = np.zeros(self.max_candidates, dtype=np.bool_)

            # 从 edge_attrs 推断偏移（近似）
            features[:num_nodes, 3] = sg.node_features
            offsets[:num_nodes] = np.zeros((num_nodes, 3), dtype=np.float32)
            mask[:num_nodes] = True

        # ---- 目标值 ----
        target = np.array([sg.center_value], dtype=np.float32)

        return {
            "candidate_features": torch.from_numpy(features),
            "candidate_offsets": torch.from_numpy(offsets),
            "candidate_mask": torch.from_numpy(mask),
            "target": torch.from_numpy(target),
            "num_nodes": sg.num_nodes,
        }


def collate_fn_c(batch):
    """
    自定义 collate 函数，处理变长子图。

    参数
    ----
    batch : List[dict]
        批次中的样本列表

    返回
    ----
    dict
        批次数据
    """
    candidate_features = torch.stack([item["candidate_features"] for item in batch])
    candidate_offsets = torch.stack([item["candidate_offsets"] for item in batch])
    candidate_mask = torch.stack([item["candidate_mask"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    num_nodes = torch.tensor([item["num_nodes"] for item in batch])

    return {
        "candidate_features": candidate_features,
        "candidate_offsets": candidate_offsets,
        "candidate_mask": candidate_mask,
        "targets": targets,
        "num_nodes": num_nodes,
    }


# ============================================================
# 5. Trainer - 训练器
# ============================================================

class Trainer:
    """
    方案C 训练器。

    支持功能：
      - 端到端联合训练（图结构学习 + GNN）
      - 分离训练（先冻结图结构，再训练 GNN）
      - 早停机制
      - 学习率调度（CosineAnnealing / StepLR）
      - 梯度裁剪
      - 模型检查点保存/加载
      - 训练日志记录

    参数
    ----
    model : AttentionGraphModel
        方案C 模型
    config : ConfigC
        配置
    """

    def __init__(self, model: AttentionGraphModel, config: ConfigC):
        self.model = model
        self.config = config
        self.device = config.get_device()
        self.model.to(self.device)

        gnn_cfg = config.gnn_model

        # ---- 优化器 ----
        # GNN 参数
        gnn_params = []
        graph_params = []

        for name, param in model.named_parameters():
            if "graph_learner" in name:
                graph_params.append(param)
            else:
                gnn_params.append(param)

        self.optimizer_gnn = torch.optim.Adam(
            gnn_params,
            lr=gnn_cfg.learning_rate,
            weight_decay=gnn_cfg.weight_decay,
        )

        # 图结构学习器使用更低的学习率
        graph_lr = gnn_cfg.learning_rate * gnn_cfg.graph_lr_ratio
        self.optimizer_graph = torch.optim.Adam(
            graph_params,
            lr=graph_lr,
            weight_decay=gnn_cfg.weight_decay,
        )

        # ---- 损失函数 ----
        self.criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        # ---- 学习率调度器 ----
        self.scheduler_gnn = None
        self.scheduler_graph = None

        if gnn_cfg.scheduler_type == "cosine":
            self.scheduler_gnn = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_gnn, T_max=gnn_cfg.num_epochs
            )
            self.scheduler_graph = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_graph, T_max=gnn_cfg.num_epochs
            )
        elif gnn_cfg.scheduler_type == "step":
            self.scheduler_gnn = torch.optim.lr_scheduler.StepLR(
                self.optimizer_gnn,
                step_size=gnn_cfg.scheduler_step_size,
                gamma=gnn_cfg.scheduler_gamma,
            )
            self.scheduler_graph = torch.optim.lr_scheduler.StepLR(
                self.optimizer_graph,
                step_size=gnn_cfg.scheduler_step_size,
                gamma=gnn_cfg.scheduler_gamma,
            )

        # ---- 训练状态 ----
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "reg_loss": [],
            "lr": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        训练一个 epoch。

        参数
        ----
        dataloader : DataLoader
            训练数据加载器

        返回
        ----
        metrics : dict
            训练指标
        """
        self.model.train()

        # 检查是否需要冻结图结构
        gnn_cfg = self.config.gnn_model
        freeze_graph = (
            gnn_cfg.freeze_graph_after_epoch > 0 and
            self.epoch >= gnn_cfg.freeze_graph_after_epoch
        )
        if freeze_graph:
            self._freeze_graph_learner()
        elif gnn_cfg.joint_training:
            self._unfreeze_graph_learner()

        total_loss = 0.0
        total_mae = 0.0
        total_reg_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # 移动数据到设备
            candidate_features = batch["candidate_features"].to(self.device)
            candidate_offsets = batch["candidate_offsets"].to(self.device)
            candidate_mask = batch["candidate_mask"].to(self.device)
            targets = batch["targets"].to(self.device)

            # ---- 前向传播 ----
            result = self.model(
                candidate_features, candidate_offsets, candidate_mask
            )
            prediction = result["prediction"]
            reg_loss = result["reg_loss"]

            # ---- 计算损失 ----
            pred_loss = self.criterion(prediction, targets)
            mae_loss = self.mae_criterion(prediction, targets)

            # 总损失 = 预测损失 + 正则化损失
            loss = pred_loss + reg_loss

            # ---- 反向传播 ----
            self.optimizer_gnn.zero_grad()
            if gnn_cfg.joint_training and not freeze_graph:
                self.optimizer_graph.zero_grad()

            loss.backward()

            # 梯度裁剪
            if gnn_cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gnn_cfg.gradient_clip
                )

            self.optimizer_gnn.step()
            if gnn_cfg.joint_training and not freeze_graph:
                self.optimizer_graph.step()

            # ---- 统计 ----
            total_loss += pred_loss.item()
            total_mae += mae_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1

        metrics = {
            "loss": total_loss / max(num_batches, 1),
            "mae": total_mae / max(num_batches, 1),
            "reg_loss": total_reg_loss / max(num_batches, 1),
        }
        return metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        验证。

        参数
        ----
        dataloader : DataLoader
            验证数据加载器

        返回
        ----
        metrics : dict
            验证指标
        """
        self.model.eval()

        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        for batch in dataloader:
            candidate_features = batch["candidate_features"].to(self.device)
            candidate_offsets = batch["candidate_offsets"].to(self.device)
            candidate_mask = batch["candidate_mask"].to(self.device)
            targets = batch["targets"].to(self.device)

            result = self.model(
                candidate_features, candidate_offsets, candidate_mask
            )
            prediction = result["prediction"]

            pred_loss = self.criterion(prediction, targets)
            mae_loss = self.mae_criterion(prediction, targets)

            total_loss += pred_loss.item()
            total_mae += mae_loss.item()
            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "mae": total_mae / max(num_batches, 1),
        }

    def fit(
        self,
        train_dataset: GraphDatasetC,
        val_dataset: Optional[GraphDatasetC] = None,
    ) -> Dict[str, List[float]]:
        """
        完整训练流程。

        参数
        ----
        train_dataset : GraphDatasetC
            训练数据集
        val_dataset : GraphDatasetC, optional
            验证数据集

        返回
        ----
        history : dict
            训练历史
        """
        gnn_cfg = self.config.gnn_model

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=gnn_cfg.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn_c,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=gnn_cfg.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=collate_fn_c,
                pin_memory=True,
            )

        logger.info(f"开始训练: epochs={gnn_cfg.num_epochs}, "
                     f"batch_size={gnn_cfg.batch_size}, "
                     f"device={self.device}")
        logger.info(f"训练集: {len(train_dataset)} 样本, "
                     f"验证集: {len(val_dataset) if val_dataset else 0} 样本")

        for epoch in range(gnn_cfg.num_epochs):
            self.epoch = epoch
            t0 = time.time()

            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证
            val_metrics = {"loss": float('inf'), "mae": float('inf')}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)

            # 学习率调度
            if self.scheduler_gnn is not None:
                self.scheduler_gnn.step()
            if self.scheduler_graph is not None:
                self.scheduler_graph.step()

            # 记录历史
            current_lr = self.optimizer_gnn.param_groups[0]['lr']
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_mae"].append(train_metrics["mae"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["reg_loss"].append(train_metrics["reg_loss"])
            self.history["lr"].append(current_lr)

            # 日志
            elapsed = time.time() - t0
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{gnn_cfg.num_epochs} "
                    f"({elapsed:.1f}s) | "
                    f"Train Loss: {train_metrics['loss']:.6f} "
                    f"MAE: {train_metrics['mae']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.6f} "
                    f"MAE: {val_metrics['mae']:.4f} | "
                    f"Reg: {train_metrics['reg_loss']:.6f} | "
                    f"LR: {current_lr:.2e}"
                )

            # 早停
            if val_loader is not None:
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0
                    self.save_checkpoint("best_model.pt", epoch)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= gnn_cfg.early_stopping_patience:
                        logger.info(f"早停触发: {epoch+1} 轮无改善")
                        break

        # 保存最终模型
        self.save_checkpoint("final_model.pt", self.epoch)
        logger.info(f"训练完成! 最佳验证损失: {self.best_val_loss:.6f}")

        return self.history

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        推理/预测。

        参数
        ----
        dataloader : DataLoader
            数据加载器

        返回
        ----
        predictions : np.ndarray
            预测值数组
        """
        self.model.eval()
        all_predictions = []

        for batch in dataloader:
            candidate_features = batch["candidate_features"].to(self.device)
            candidate_offsets = batch["candidate_offsets"].to(self.device)
            candidate_mask = batch["candidate_mask"].to(self.device)

            result = self.model(
                candidate_features, candidate_offsets, candidate_mask
            )
            predictions = result["prediction"].cpu().numpy()
            all_predictions.append(predictions)

        return np.concatenate(all_predictions, axis=0)

    def save_checkpoint(self, filename: str, epoch: int):
        """
        保存模型检查点。

        参数
        ----
        filename : str
            文件名
        epoch : int
            当前 epoch
        """
        path = os.path.join(self.config.model_save_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_gnn_state_dict": self.optimizer_gnn.state_dict(),
            "optimizer_graph_state_dict": self.optimizer_graph.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }, path)
        logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, filename: str):
        """
        加载模型检查点。

        参数
        ----
        filename : str
            文件名
        """
        path = os.path.join(self.config.model_save_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点文件不存在: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_gnn.load_state_dict(checkpoint["optimizer_gnn_state_dict"])
        self.optimizer_graph.load_state_dict(checkpoint["optimizer_graph_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint.get("history", self.history)
        self.epoch = checkpoint["epoch"]
        logger.info(f"检查点已加载: {path} (epoch={self.epoch})")

    def _freeze_graph_learner(self):
        """冻结图结构学习器"""
        for param in self.model.graph_learner.parameters():
            param.requires_grad = False

    def _unfreeze_graph_learner(self):
        """解冻图结构学习器"""
        for param in self.model.graph_learner.parameters():
            param.requires_grad = True
