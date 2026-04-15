"""
ntl_graph_accel v2 - Numba JIT 加速版
=====================================
重构目标：
  1. 移除无效的 GPU Bresenham，改用 Numba JIT 编译核心计算
  2. 消除 graph_builder 中的 Python for 循环和动态数据结构
  3. 保留缓存复用 + 空间分块并行（已验证有效）

v1 → v2 性能提升来源：
  - _select_nodes: Python 循环 → Numba 向量化（~10x）
  - _build_edges:  Python 循环 + dict → Numba JIT（~20x）
  - _bresenham:    查找表（保留）+ Numba 路径过滤（~5x）
"""

__version__ = "2.0.0"
