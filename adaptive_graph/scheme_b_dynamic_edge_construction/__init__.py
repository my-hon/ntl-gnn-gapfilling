"""
方案B：空间异质性感知的动态连边
================================

灵感来源：
  - Graph WaveNet (arXiv:1906.00121) 的自适应邻接矩阵
  - DRTR (arXiv:2406.17281) 的距离感知拓扑精炼

核心思想：
  根据空间异质性（NTL值的空间变异系数）动态调整边的连接策略——
  高异质性区域（城市核心）增加空间维度权重，
  低异质性区域（郊区）增加时序维度权重。

模块结构：
  - config_b.py              : 方案B配置（含异质性参数）
  - heterogeneity_analyzer.py: 空间异质性分析器（Numba JIT 加速）
  - dynamic_edge_builder.py  : 动态边构建器（Numba JIT 加速）
  - graph_builder_b.py       : 方案B图构建器（与 v2 SubGraph 接口兼容）

使用示例：
    >>> from adaptive_graph.scheme_b_dynamic_edge_construction import (
    ...     ConfigB, GraphBuilderB
    ... )
    >>> config = ConfigB()
    >>> builder = GraphBuilderB(config, data, valid_mask)
    >>> subgraph = builder.build_single(tc, hc, wc)
    >>> print(f"异质性指数: {subgraph.heterogeneity_index:.3f}")
    >>> print(f"空间权重: {subgraph.w_spatial:.3f}")
"""

__version__ = "1.0.0"

from .config_b import ConfigB, HeterogeneityConfig
from .heterogeneity_analyzer import HeterogeneityAnalyzer
from .dynamic_edge_builder import DynamicEdgeBuilder
from .graph_builder_b import GraphBuilderB, SubGraph

__all__ = [
    'ConfigB',
    'HeterogeneityConfig',
    'HeterogeneityAnalyzer',
    'DynamicEdgeBuilder',
    'GraphBuilderB',
    'SubGraph',
]
