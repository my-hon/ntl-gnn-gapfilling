"""
全局配置参数模块（v2）
=====================
与参考实现 build_dataset.py 对齐。
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class DataConfig:
    """数据相关配置"""
    data_shape: Tuple[int, int, int] = (366, 476, 520)
    nan_value: float = float('nan')
    feature_scale: float = 100.0       # 归一化因子 (/10.0 恢复辐射, /100.0 归一化)
    edge_scale: float = 8.0            # 边属性归一化因子
    edge_time: int = 50                # 有效区域时间缓冲
    edge_height: int = 50              # 有效区域高度缓冲
    edge_width: int = 50               # 有效区域宽度缓冲
    quality_path: str = ""             # 质量标志文件路径


@dataclass
class GraphConfig:
    """图构建相关配置"""
    search_node: int = 32              # 邻居节点数 (总节点 = search_node + 1 中心)
    num_nodes: int = 33                # search_node + 1 (中心)
    ext_range: int = 6                 # 子立方体提取范围 (EXT_RANGE)
    max_ext: int = 20                  # 最大自适应扩展
    num_regions: int = 6               # 象限数量
    natural_breaks: List[float] = field(default_factory=lambda: [
        -float('inf'), 0.001, 0.00325, 0.0065, 0.0125,
        0.025, 0.1, float('inf')
    ])
    sample_per_class: int = 20000      # 每个自然断点类别的采样数


@dataclass
class AccelerationConfig:
    """加速策略配置（v2: Numba JIT + 查找表 + 缓存 + 分块并行）"""
    # ---- 空间分块并行 ----
    tile_size: int = 128
    num_workers: int = 8

    # ---- Numba JIT ----
    use_numba: bool = True

    # ---- Bresenham 查找表 ----
    bresenham_lookup: bool = True

    # ---- 缓存复用 ----
    use_cache: bool = True
    cache_quantization: int = 2
    cache_max_size: int = 50000

    # ---- 输出 ----
    output_format: str = "pkl"
    save_per_tile: bool = True


@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    accel: AccelerationConfig = field(default_factory=AccelerationConfig)

    input_path: str = ""
    output_dir: str = "./output_graphs_v2"
    cache_dir: str = "./graph_cache_v2"
    seed: int = 0

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        # 保持 num_nodes 与 search_node 同步
        self.graph.num_nodes = self.graph.search_node + 1

    def get_effective_temporal_range(self):
        buf = self.data.edge_time
        T = self.data.data_shape[0]
        return buf, T - buf

    def get_effective_spatial_range(self):
        h_buf = self.data.edge_height
        w_buf = self.data.edge_width
        H, W = self.data.data_shape[1], self.data.data_shape[2]
        return h_buf, H - h_buf, w_buf, W - w_buf
