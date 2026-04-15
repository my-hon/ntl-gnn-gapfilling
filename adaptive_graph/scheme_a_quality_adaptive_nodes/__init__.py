"""
方案A：数据质量驱动的自适应节点数
====================================
灵感来源：
  - Beyond kNN: Graph Neural Networks for Point Clouds with Adaptive Neighborhood (arXiv:2208.00604)
    → 最优传输自适应邻域思想
  - Pro-GNN: Property Graph Neural Network (arXiv:2005.10203)
    → 稀疏约束思想

核心思想：
  根据每个位置邻域内的数据质量（有效像素密度、空间连续性、时序稳定性），
  动态调整图中节点数量，而非固定36个。

模块说明：
  - config_a.py          : 方案A配置（继承v2，增加质量自适应参数）
  - adaptive_selector.py : 自适应节点数选择器（Numba JIT加速核心计算）
  - graph_builder_a.py   : 方案A图构建器（继承v2 GraphBuilder逻辑，重写节点选择）
"""

__version__ = "1.0.0"

from .config_a import ConfigA, GraphConfigA, QualityAdaptiveConfig
from .adaptive_selector import AdaptiveNodeSelector
from .graph_builder_a import GraphBuilderA

__all__ = [
    'ConfigA',
    'GraphConfigA',
    'QualityAdaptiveConfig',
    'AdaptiveNodeSelector',
    'GraphBuilderA',
]
