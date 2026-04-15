"""
方案B配置模块 - 空间异质性感知的动态连边
==========================================
在 v2 基础配置之上，增加空间异质性相关参数。

灵感来源：
  - Graph WaveNet (arXiv:1906.00121) 的自适应邻接矩阵
  - DRTR (arXiv:2406.17281) 的距离感知拓扑精炼

核心思想：
  根据空间异质性（NTL值的空间变异系数）动态调整边的连接策略——
  高异质性区域（城市核心）增加空间维度权重，
  低异质性区域（郊区）增加时序维度权重。
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

# 复用 v2 的基础配置类型
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ntl_graph_accel_v2.config import DataConfig, GraphConfig, AccelerationConfig


@dataclass
class HeterogeneityConfig:
    """空间异质性相关配置

    参数说明：
      h_threshold : float
          异质性阈值 H_threshold。当 H > h_threshold 时判定为高异质性区域。
          典型值范围 [0.1, 0.5]，默认 0.25。

      h_scale : float
          sigmoid 缩放因子 scale，控制 w_spatial 对 H 的敏感度。
          值越大，权重变化越陡峭。默认 10.0。

      high_het_spatial_boost : float
          高异质性区域的 Bresenham 截断放松系数。
          >1.0 表示允许更长的空间路径（放松截断条件），默认 1.5。

      low_het_temporal_extend : int
          低异质性区域的时序窗口扩展步数。
          >0 表示允许连接更远时间步的节点，默认 3。

      heterogeneity_cube_radius : int
          计算异质性指数时使用的局部立方体半径。
          决定了空间异质性的感受野大小，默认 5。

      min_valid_for_heterogeneity : int
          计算异质性指数所需的最小有效像素数。
          低于此值时使用默认权重，默认 8。
    """
    h_threshold: float = 0.25
    h_scale: float = 10.0
    high_het_spatial_boost: float = 1.5
    low_het_temporal_extend: int = 3
    heterogeneity_cube_radius: int = 5
    min_valid_for_heterogeneity: int = 8


@dataclass
class ConfigB:
    """方案B总配置

    与 v2 Config 兼容，额外包含异质性配置。
    可直接传入 GraphBuilderB 使用。
    """
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    accel: AccelerationConfig = field(default_factory=AccelerationConfig)
    heterogeneity: HeterogeneityConfig = field(default_factory=HeterogeneityConfig)

    input_path: str = ""
    output_dir: str = "./output_graphs_b"
    cache_dir: str = "./graph_cache_b"
    seed: int = 0

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_effective_temporal_range(self):
        """获取有效时间范围（去除缓冲区）"""
        T = self.data.data_shape[0]
        buf = min(self.data.temporal_buffer, T // 2)
        return buf, T - buf

    def get_effective_spatial_range(self):
        """获取有效空间范围（去除缓冲区）"""
        buf = min(self.data.buffer_size,
                  min(self.data.data_shape[1], self.data.data_shape[2]) // 2)
        H, W = self.data.data_shape[1], self.data.data_shape[2]
        return buf, H - buf, buf, W - buf
