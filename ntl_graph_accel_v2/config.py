"""
全局配置参数模块（v2）
=====================
与 v1 配置兼容，移除了无效的 GPU 相关配置。
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    """数据相关配置"""
    data_shape: Tuple[int, int, int] = (366, 476, 520)
    buffer_size: int = 50
    temporal_buffer: int = 10
    nan_value: float = float('nan')
    feature_scale: float = 108.3
    edge_scale: float = 8.0


@dataclass
class GraphConfig:
    """图构建相关配置"""
    num_nodes: int = 36
    initial_radius: int = 4
    max_radius: int = 20
    num_regions: int = 6
    max_bresenham_len: int = 30


@dataclass
class AccelerationConfig:
    """加速策略配置（v2: Numba JIT + 查找表 + 缓存 + 分块并行）"""
    # ---- 空间分块并行 ----
    tile_size: int = 128
    num_workers: int = 8

    # ---- Numba JIT ----
    use_numba: bool = True        # 是否启用 Numba JIT 编译

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

    def get_effective_temporal_range(self):
        T = self.data.data_shape[0]
        buf = min(self.data.temporal_buffer, T // 2)
        return buf, T - buf

    def get_effective_spatial_range(self):
        buf = min(self.data.buffer_size, min(self.data.data_shape[1], self.data.data_shape[2]) // 2)
        H, W = self.data.data_shape[1], self.data.data_shape[2]
        return buf, H - buf, buf, W - buf
