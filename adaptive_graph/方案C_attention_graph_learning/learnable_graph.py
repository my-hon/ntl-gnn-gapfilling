"""
方案C：可学习图结构模块
========================
基于 PyTorch 的可学习图结构学习，包含三个核心组件：
  1. LearnableNodeSelector: MLP 节点选择器，从候选池中选择 top-K 节点
  2. LearnableEdgeBuilder: 基于特征相似度的可学习边权重构建
  3. GraphStructureLearner: 整合节点选择 + 边构建的端到端图结构学习

灵感来源：
  - Pro-GNN (arXiv:2005.10203): 可学习邻接矩阵
  - ASTGCN (AAAI 2019): 空间注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


# ============================================================
# 工具函数
# ============================================================

def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> nn.Sequential:
    """
    构建多层感知机（MLP）。

    参数
    ----
    input_dim : int
        输入维度
    hidden_dim : int
        隐藏层维度
    output_dim : int
        输出维度
    num_layers : int
        隐藏层数量（不含输入/输出层）
    activation : str
        激活函数类型: "relu" / "gelu" / "tanh" / "leaky_relu"
    dropout : float
        Dropout 比率
    batch_norm : bool
        是否使用 BatchNorm

    返回
    ----
    nn.Sequential
        构建好的 MLP 模型
    """
    # 选择激活函数
    act_map = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "leaky_relu": lambda: nn.LeakyReLU(0.2),
    }
    act_cls = act_map.get(activation, nn.ReLU)

    layers = []
    in_dim = input_dim

    # 隐藏层
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    # 输出层
    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)


# ============================================================
# 1. LearnableNodeSelector - MLP 节点选择器
# ============================================================

class LearnableNodeSelector(nn.Module):
    """
    可学习节点选择器。

    从候选节点池中选择最重要的 top-K 个节点。
    每个候选节点的输入特征为 (dt, dh, dw, value)，
    通过 MLP 计算重要性分数，选择分数最高的 K 个节点。

    与 v2 的区别：
      - v2: 固定按距离排序 + 区域配额选择
      - 方案C: MLP 学习节点重要性，自适应选择

    参数
    ----
    input_dim : int
        输入特征维度（默认4: dt, dh, dw, value）
    hidden_dim : int
        MLP 隐藏层维度
    output_dim : int
        注意力分数投影维度
    num_layers : int
        MLP 层数
    top_k : int
        选择的节点数量
    temperature : float
        softmax 温度参数（控制选择分布的锐度）
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 16,
        num_layers: int = 2,
        top_k: int = 36,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.top_k = top_k
        self.temperature = temperature

        # 重要性评分 MLP: 输入 -> 隐藏 -> 1（标量分数）
        self.score_mlp = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            activation="gelu",
            dropout=0.1,
        )

        # 可学习的位置编码（用于捕获空间先验）
        # 将 (dt, dh, dw) 归一化后编码
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # 位置编码与原始特征的融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # 融合后的评分头
        self.fused_score_head = nn.Linear(hidden_dim, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        """参数初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_offsets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：从候选节点中选择 top-K。

        参数
        ----
        candidate_features : torch.Tensor
            候选节点特征，形状 (batch_size, num_candidates, input_dim)
            最后一维为 (dt, dh, dw, value)
        candidate_offsets : torch.Tensor
            候选节点偏移量，形状 (batch_size, num_candidates, 3)
            最后一维为 (dt, dh, dw)
        mask : torch.Tensor, optional
            有效候选节点掩码，形状 (batch_size, num_candidates)
            True 表示有效，False 表示无效

        返回
        ----
        selected_indices : torch.Tensor
            选中的节点索引，形状 (batch_size, top_k)
        selected_scores : torch.Tensor
            选中节点的分数，形状 (batch_size, top_k)
        attention_weights : torch.Tensor
            所有候选节点的注意力权重，形状 (batch_size, num_candidates)
        """
        batch_size, num_candidates, _ = candidate_features.shape

        # ---- 1. 计算位置编码 ----
        # 归一化偏移量到 [-1, 1]
        max_radius = candidate_offsets.abs().max().clamp(min=1.0)
        normalized_offsets = candidate_offsets / max_radius

        pos_encoding = self.position_encoder(normalized_offsets)  # (B, N, hidden_dim)

        # ---- 2. 特征融合 ----
        # 拼接位置编码和原始特征
        combined = torch.cat([
            pos_encoding,
            candidate_features
        ], dim=-1)  # (B, N, hidden_dim + input_dim)

        gate = self.fusion_gate(combined)  # (B, N, hidden_dim)
        fused = gate * pos_encoding  # (B, N, hidden_dim)

        # ---- 3. 计算重要性分数 ----
        scores = self.fused_score_head(fused).squeeze(-1)  # (B, N)

        # ---- 4. 应用掩码（无效候选节点分数设为 -inf）----
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # ---- 5. 温度缩放的 softmax ----
        attention_weights = F.softmax(scores / self.temperature, dim=-1)  # (B, N)

        # ---- 6. 选择 top-K ----
        k = min(self.top_k, num_candidates)
        selected_scores, selected_indices = torch.topk(scores, k=k, dim=-1)

        return selected_indices, selected_scores, attention_weights

    def get_selection_loss(
        self,
        attention_weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算节点选择的正则化损失。
        鼓励注意力分布更加集中（熵正则化）。

        参数
        ----
        attention_weights : torch.Tensor
            注意力权重，形状 (batch_size, num_candidates)
        mask : torch.Tensor, optional
            有效节点掩码

        返回
        ----
        loss : torch.Tensor
            熵正则化损失（标量）
        """
        # 计算注意力分布的熵
        log_weights = torch.log(attention_weights + 1e-10)
        entropy = -(attention_weights * log_weights).sum(dim=-1)  # (B,)

        if mask is not None:
            # 只对有效节点计算
            num_valid = mask.sum(dim=-1).clamp(min=1).float()
            max_entropy = torch.log(num_valid)
            normalized_entropy = entropy / max_entropy
        else:
            num_candidates = attention_weights.shape[-1]
            normalized_entropy = entropy / math.log(num_candidates)

        # 最小化熵（鼓励集中选择）
        return normalized_entropy.mean()


# ============================================================
# 2. LearnableEdgeBuilder - 可学习边构建器
# ============================================================

class LearnableEdgeBuilder(nn.Module):
    """
    基于特征相似度的可学习边权重构建器。

    替代 v2 中的 Bresenham 硬连接，使用节点特征相似度计算软边权重。
    支持 KNN + 全连接混合策略。

    与 v2 的区别：
      - v2: Bresenham 3D 射线检测遮挡关系，硬连接
      - 方案C: 基于特征相似度的注意力边权重，软连接

    参数
    ----
    feature_dim : int
        节点特征维度
    hidden_dim : int
        MLP 隐藏层维度
    knn_k : int
        KNN 邻居数
    self_loop : bool
        是否添加自环
    symmetric : bool
        是否构建对称边
    """

    def __init__(
        self,
        feature_dim: int = 1,
        hidden_dim: int = 32,
        knn_k: int = 4,
        self_loop: bool = True,
        symmetric: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.knn_k = knn_k
        self.self_loop = self_loop
        self.symmetric = symmetric

        # 边权重计算 MLP
        # 输入: 两节点特征的拼接 + 距离特征
        edge_input_dim = 2 * feature_dim + 3  # 特征拼接 + (dt, dh, dw) 距离
        self.edge_weight_mlp = build_mlp(
            input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=2,
            activation="gelu",
            dropout=0.1,
        )

        # 可学习的相似度度量参数
        # 将节点特征投影到度量空间
        self.feature_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 距离编码
        self.distance_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
        )

        # 注意力边权重计算
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """参数初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        node_offsets: torch.Tensor,
        selected_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：构建可学习的边。

        参数
        ----
        node_features : torch.Tensor
            节点特征，形状 (batch_size, num_nodes, feature_dim)
        node_offsets : torch.Tensor
            节点偏移量，形状 (batch_size, num_nodes, 3)
        selected_indices : torch.Tensor, optional
            选中的节点索引，形状 (batch_size, top_k)
            如果提供，则只对选中的节点构建边

        返回
        ----
        adj_matrix : torch.Tensor
            邻接矩阵，形状 (batch_size, num_nodes, num_nodes)
        edge_weights : torch.Tensor
            边权重，形状 (batch_size, num_edges)
        edge_index : torch.Tensor
            边索引，形状 (2, batch_num_edges)
        """
        batch_size, num_nodes, feat_dim = node_features.shape

        # 如果提供了选中索引，则只处理选中的节点
        if selected_indices is not None:
            # 收集选中的节点特征和偏移
            expanded_indices = selected_indices.unsqueeze(-1).expand(
                -1, -1, feat_dim
            )  # (B, K, feat_dim)
            selected_features = torch.gather(
                node_features, 1, expanded_indices
            )  # (B, K, feat_dim)

            expanded_offsets = selected_indices.unsqueeze(-1).expand(
                -1, -1, 3
            )  # (B, K, 3)
            selected_offsets = torch.gather(
                node_offsets, 1, expanded_offsets
            )  # (B, K, 3)

            num_selected = selected_indices.shape[1]
        else:
            selected_features = node_features
            selected_offsets = node_offsets
            num_selected = num_nodes

        # ---- 1. 投影特征到度量空间 ----
        projected = self.feature_projector(selected_features)  # (B, K, hidden_dim)

        # ---- 2. 计算节点间距离 ----
        # 偏移量差值
        diff_offsets = (
            selected_offsets.unsqueeze(2) - selected_offsets.unsqueeze(1)
        )  # (B, K, K, 3)
        distances = diff_offsets.float().norm(dim=-1, p=2)  # (B, K, K)

        # ---- 3. 计算特征相似度（余弦相似度）----
        # 归一化
        projected_norm = F.normalize(projected, p=2, dim=-1)  # (B, K, hidden_dim)
        similarity = torch.bmm(
            projected_norm, projected_norm.transpose(1, 2)
        )  # (B, K, K)

        # ---- 4. 编码距离信息 ----
        # 归一化距离
        max_dist = distances.max().clamp(min=1.0)
        normalized_dist = distances / max_dist  # (B, K, K)
        dist_encoding = self.distance_encoder(
            diff_offsets.float().reshape(batch_size * num_selected * num_selected, 3)
        ).reshape(batch_size, num_selected, num_selected, -1)  # (B, K, K, hidden//2)

        # ---- 5. 计算注意力边权重 ----
        # 拼接: 源节点特征 + 目标节点特征 + 距离编码
        src_proj = projected.unsqueeze(2).expand(
            -1, -1, num_selected, -1
        )  # (B, K, K, hidden_dim)
        dst_proj = projected.unsqueeze(1).expand(
            -1, num_selected, -1, -1
        )  # (B, K, K, hidden_dim)

        combined = torch.cat([
            src_proj, dst_proj, dist_encoding
        ], dim=-1)  # (B, K, K, hidden_dim * 2 + hidden//2)

        edge_attention = self.attention_mlp(combined).squeeze(-1)  # (B, K, K)

        # ---- 6. 融合相似度和注意力 ----
        # 最终边权重 = sigmoid(注意力分数) * 相似度
        edge_weights_full = torch.sigmoid(edge_attention) * (0.5 + 0.5 * similarity)

        # ---- 7. 构建 KNN 稀疏邻接矩阵 ----
        k = min(self.knn_k, num_selected - 1)
        if k > 0:
            # 对每个节点，保留相似度最高的 K 个邻居
            _, knn_indices = torch.topk(similarity, k=k + 1, dim=-1)  # +1 包含自身
            # 创建 KNN 掩码
            knn_mask = torch.zeros_like(similarity, dtype=torch.bool)
            batch_range = torch.arange(batch_size, device=similarity.device).unsqueeze(1).unsqueeze(2)
            node_range = torch.arange(num_selected, device=similarity.device).unsqueeze(0).unsqueeze(2)
            knn_mask.scatter_(
                1,
                knn_indices.unsqueeze(2).expand(-1, -1, num_selected),
                True,
            )
            # 应用 KNN 掩码
            edge_weights_full = edge_weights_full * knn_mask.float()

        # ---- 8. 添加自环 ----
        if self.self_loop:
            eye = torch.eye(num_selected, device=edge_weights_full.device).unsqueeze(0)
            edge_weights_full = edge_weights_full + eye

        # ---- 9. 对称化 ----
        if self.symmetric:
            edge_weights_full = (edge_weights_full + edge_weights_full.transpose(1, 2)) / 2

        # ---- 10. 构建稀疏边索引 ----
        edge_index_list = []
        edge_weight_list = []
        for b in range(batch_size):
            adj = edge_weights_full[b]  # (K, K)
            # 提取非零边
            nonzero_mask = adj > 1e-6
            src_idx, dst_idx = torch.where(nonzero_mask)
            weights = adj[src_idx, dst_idx]
            # 添加 batch 偏移（用于批处理）
            edge_index_list.append(torch.stack([src_idx, dst_idx], dim=0))
            edge_weight_list.append(weights)

        return edge_weights_full, edge_weight_list, edge_index_list

    def get_sparsity_loss(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        计算稀疏正则化损失。
        鼓励邻接矩阵更加稀疏，防止过拟合。

        参数
        ----
        adj_matrix : torch.Tensor
            邻接矩阵，形状 (batch_size, num_nodes, num_nodes)

        返回
        ----
        loss : torch.Tensor
            稀疏正则化损失（Frobenius 范数）
        """
        # L1 范数鼓励稀疏性
        return adj_matrix.norm(p=1, dim=(-2, -1)).mean()

    def get_smoothness_loss(
        self,
        adj_matrix: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算平滑正则化损失。
        鼓励相邻节点具有相似的特征（图拉普拉斯正则化）。

        参数
        ----
        adj_matrix : torch.Tensor
            邻接矩阵，形状 (batch_size, num_nodes, num_nodes)
        node_features : torch.Tensor
            节点特征，形状 (batch_size, num_nodes, feature_dim)

        返回
        ----
        loss : torch.Tensor
            平滑正则化损失
        """
        # 图拉普拉斯: L = D - A
        degree = adj_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        # Tr(X^T L X) / |V| = sum_{i,j} A_{ij} ||x_i - x_j||^2 / |V|
        diff = node_features.unsqueeze(2) - node_features.unsqueeze(1)  # (B, N, N, F)
        smoothness = (adj_matrix.unsqueeze(-1) * diff.pow(2)).sum(dim=(-3, -2))
        return smoothness.mean()


# ============================================================
# 3. GraphStructureLearner - 图结构学习器
# ============================================================

class GraphStructureLearner(nn.Module):
    """
    端到端图结构学习器。

    整合 LearnableNodeSelector 和 LearnableEdgeBuilder，
    实现图结构学习 + GNN 训练的联合优化。

    工作流程：
      1. 输入候选节点池的特征和偏移
      2. LearnableNodeSelector 选择 top-K 节点
      3. LearnableEdgeBuilder 构建可学习的边权重
      4. 输出优化后的邻接矩阵和边索引

    参数
    ----
    node_selector_cfg : dict
        LearnableNodeSelector 配置参数
    edge_builder_cfg : dict
        LearnableEdgeBuilder 配置参数
    edge_threshold : float
        边权重阈值（低于此值的边被移除）
    sparsity_weight : float
        稀疏正则化权重
    smoothness_weight : float
        平滑正则化权重
    """

    def __init__(
        self,
        node_selector_cfg: Optional[Dict] = None,
        edge_builder_cfg: Optional[Dict] = None,
        edge_threshold: float = 0.05,
        sparsity_weight: float = 0.01,
        smoothness_weight: float = 0.005,
    ):
        super().__init__()

        # 默认配置
        ns_cfg = node_selector_cfg or {}
        eb_cfg = edge_builder_cfg or {}

        # ---- 节点选择器 ----
        self.node_selector = LearnableNodeSelector(
            input_dim=ns_cfg.get("input_dim", 4),
            hidden_dim=ns_cfg.get("hidden_dim", 64),
            output_dim=ns_cfg.get("output_dim", 16),
            num_layers=ns_cfg.get("num_layers", 2),
            top_k=ns_cfg.get("top_k", 36),
            temperature=ns_cfg.get("temperature", 1.0),
        )

        # ---- 边构建器 ----
        self.edge_builder = LearnableEdgeBuilder(
            feature_dim=eb_cfg.get("feature_dim", 1),
            hidden_dim=eb_cfg.get("hidden_dim", 32),
            knn_k=eb_cfg.get("knn_k", 4),
            self_loop=eb_cfg.get("self_loop", True),
            symmetric=eb_cfg.get("symmetric", True),
        )

        # ---- 正则化参数 ----
        self.edge_threshold = edge_threshold
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_offsets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播：完整的图结构学习流程。

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
            包含以下键：
            - "selected_indices": 选中的节点索引 (B, K)
            - "selected_scores": 选中节点的分数 (B, K)
            - "attention_weights": 注意力权重 (B, num_candidates)
            - "adj_matrix": 邻接矩阵 (B, K, K)
            - "edge_weights": 边权重列表
            - "edge_index": 边索引列表
            - "selected_features": 选中节点的特征 (B, K, feat_dim)
            - "selected_offsets": 选中节点的偏移 (B, K, 3)
            - "reg_loss": 正则化损失
        """
        # ---- 1. 节点选择 ----
        selected_indices, selected_scores, attention_weights = self.node_selector(
            candidate_features, candidate_offsets, mask
        )

        # ---- 2. 收集选中节点的特征 ----
        batch_size, num_candidates, feat_dim = candidate_features.shape
        k = selected_indices.shape[1]

        # 收集特征
        expanded_indices = selected_indices.unsqueeze(-1).expand(-1, -1, feat_dim)
        selected_features = torch.gather(candidate_features, 1, expanded_indices)

        # 收集偏移
        expanded_offsets = selected_indices.unsqueeze(-1).expand(-1, -1, 3)
        selected_offsets = torch.gather(candidate_offsets, 1, expanded_offsets)

        # ---- 3. 边构建 ----
        # 只使用 value 作为节点特征（第4维）
        node_feat_for_edge = selected_features[:, :, -1:]  # (B, K, 1)
        adj_matrix, edge_weights, edge_index = self.edge_builder(
            node_feat_for_edge, selected_offsets, selected_indices
        )

        # ---- 4. 计算正则化损失 ----
        reg_loss = torch.tensor(0.0, device=candidate_features.device)

        # 节点选择熵正则化
        entropy_loss = self.node_selector.get_selection_loss(
            attention_weights, mask
        )
        reg_loss = reg_loss + entropy_loss

        # 边稀疏正则化
        if self.sparsity_weight > 0:
            sparsity_loss = self.edge_builder.get_sparsity_loss(adj_matrix)
            reg_loss = reg_loss + self.sparsity_weight * sparsity_loss

        # 特征平滑正则化
        if self.smoothness_weight > 0:
            smoothness_loss = self.edge_builder.get_smoothness_loss(
                adj_matrix, node_feat_for_edge
            )
            reg_loss = reg_loss + self.smoothness_weight * smoothness_loss

        return {
            "selected_indices": selected_indices,
            "selected_scores": selected_scores,
            "attention_weights": attention_weights,
            "adj_matrix": adj_matrix,
            "edge_weights": edge_weights,
            "edge_index": edge_index,
            "selected_features": selected_features,
            "selected_offsets": selected_offsets,
            "reg_loss": reg_loss,
        }

    def get_adjacency_for_gnn(
        self,
        result: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 forward 的输出中提取 GNN 可用的邻接矩阵。

        参数
        ----
        result : dict
            forward 方法的输出

        返回
        ----
        adj_matrix : torch.Tensor
            阈值化后的邻接矩阵 (B, K, K)
        edge_weights : torch.Tensor
            归一化的边权重
        """
        adj = result["adj_matrix"]

        # 应用阈值
        adj = adj * (adj > self.edge_threshold).float()

        # 行归一化（D^{-1} A）
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        adj_normalized = adj / degree

        return adj, adj_normalized
