"""
NTL图加速构建工具包
==================
基于GNN的夜间灯光数据缺失值填补 - 子图生成加速框架。

模块说明:
  config.py              - 全局配置参数
  data_loader.py         - 数据加载与预处理
  gpu_bresenham.py       - GPU加速3D Bresenham算法
  graph_builder.py       - 核心图构建逻辑
  graph_cache.py         - 缓存复用机制
  spatial_partitioner.py - 空间分块并行化
  main.py                - 主控调度入口

使用方法:
  # 缺失值填补模式
  python -m ntl_graph_accel.main --mode missing --input data.npy --output ./output

  # 训练数据构建模式
  python -m ntl_graph_accel.main --mode training --input data.npy --output ./output
"""

__version__ = "1.0.0"
__author__ = "NTL-GNN Research"
