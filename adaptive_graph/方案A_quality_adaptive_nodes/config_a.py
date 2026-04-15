"""
方案A配置模块 - 数据质量驱动的自适应节点数
============================================
在 v2 Config 基础上，增加质量自适应相关参数。

灵感来源：
  - Beyond kNN (arXiv:2208.00604) 的最优传输自适应邻域
  - Pro-GNN (arXiv:2005.10203) 的稀疏约束
"""

from dataclasses import dataclass, field
from typing import Tuple

# 复用 v2 的基础配置
from ntl_graph_accel_v2.config import (
    DataConfig,
    AccelerationConfig,
    Config as V2Config,
)


@dataclass
class QualityAdaptiveConfig:
    """
    数据质量自适应参数。

    质量指标定义：
      - 有效像素密度 rho:   邻域内有效像素占比，范围 [0, 1]
      - 空间连续性 S:       邻域内有效像素的空间方差（归一化），范围 [0, 1]
      - 时序稳定性 T:       时间维度标准差/均值（归一化），范围 [0, 1]

    综合质量分数: Q = rho^alpha * S^beta * T^gamma
    自适应节点数: N = clip(N_base * f(Q), N_min, N_max)
                  其中 f(Q) = 0.5 + Q
    """
    # ---- 质量指标权重 ----
    alpha: float = 0.5    # 有效像素密度权重
    beta: float = 0.3     # 空间连续性权重
    gamma: float = 0.2    # 时序稳定性权重

    # ---- 节点数范围 ----
    n_min: int = 18       # 最小节点数（N_base/2）
    n_max: int = 54       # 最大节点数（N_base*1.5）

    # ---- 质量分数缩放函数 ----
    # f(Q) = scale_offset + scale_slope * Q
    # 默认: f(Q) = 0.5 + Q, 即 Q=0 时 N=N_base/2, Q=1 时 N=N_base*1.5
    scale_offset: float = 0.5
    scale_slope: float = 1.0

    # ---- 积分图参数（用于高效计算局部质量） ----
    integral_window: int = 5   # 积分图滑动窗口半径

    # ---- 时序稳定性计算参数 ----
    temporal_window: int = 3   # 时序窗口半径（前后各取几帧）


@dataclass
class GraphConfigA:
    """
    方案A的图构建配置。

    在 v2 GraphConfig 基础上，num_nodes 变为基准值（N_base），
    实际节点数由质量自适应模块动态决定。
    """
    num_nodes_base: int = 36       # 基准节点数 N_base
    initial_radius: int = 4
    max_radius: int = 20
    num_regions: int = 6
    max_bresenham_len: int = 30


@dataclass
class ConfigA:
    """
    方案A总配置。

    与 v2 Config 兼容，增加 quality_adaptive 字段。
    """
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfigA = field(default_factory=GraphConfigA)
    accel: AccelerationConfig = field(default_factory=AccelerationConfig)
    quality: QualityAdaptiveConfig = field(default_factory=QualityAdaptiveConfig)

    input_path: str = ""
    output_dir: str = "./output_graphs_a"
    cache_dir: str = "./graph_cache_a"
    seed: int = 0

    def __post_init__(self):
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_effective_temporal_range(self):
        T = self.data.data_shape[0]
        buf = min(self.data.temporal_buffer, T // 2)
        return buf, T - buf

    def get_effective_spatial_range(self):
        buf = min(self.data.buffer_size,
                  min(self.data.data_shape[1], self.data.data_shape[2]) // 2)
        H, W = self.data.data_shape[1], self.data.data_shape[2]
        return buf, H - buf, buf, W - buf

    def to_v2_config(self):
        """
        转换为 v2 Config（用于复用 v2 的 GraphBuilder）。

        注意：search_node 使用基准值，实际自适应在 GraphBuilderA 中处理。
        """
        from ntl_graph_accel_v2.config import GraphConfig
        return V2Config(
            data=self.data,
            graph=GraphConfig(
                search_node=self.graph.num_nodes_base,
                ext_range=self.graph.initial_radius,
                max_ext=self.graph.max_radius,
                num_regions=self.graph.num_regions,
            ),
            accel=self.accel,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            seed=self.seed,
        )
