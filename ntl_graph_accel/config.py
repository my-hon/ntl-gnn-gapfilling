"""
全局配置参数模块
================
集中管理所有图构建与加速相关的超参数。
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, List


@dataclass
class DataConfig:
    """数据相关配置"""
    # 数据形状 (T, H, W)
    data_shape: Tuple[int, int, int] = (366, 476, 520)
    # 缓冲区大小（像素）
    buffer_size: int = 50
    # 有效数据区域（去除缓冲区后的范围）
    # 时间维度：去除前后10天缓冲
    temporal_buffer: int = 10
    # 数据质量标记：NaN表示缺失/低质量
    nan_value: float = float('nan')
    # 节点特征归一化因子（99.9百分位）
    feature_scale: float = 108.3
    # 边属性归一化因子
    edge_scale: float = 8.0
    # 时空子立方体提取范围（替代 initial_radius）
    ext_range: int = 6
    # 搜索节点数
    search_node: int = 32
    # 自然断点分箱边界
    natural_breaks: List[float] = field(default_factory=lambda: [
        float('-inf'), 0.001, 0.00325, 0.0065, 0.0125,
        0.025, 0.1, float('inf')
    ])
    # 每个类别采样数
    sample_per_class: int = 20000
    # 边属性中时间维度的缩放因子
    edge_time: int = 50
    # 边属性中高度维度的缩放因子
    edge_height: int = 50
    # 边属性中宽度维度的缩放因子
    edge_width: int = 50
    # 数据质量文件路径
    quality_path: str = ""


@dataclass
class GraphConfig:
    """图构建相关配置"""
    # 图中节点总数（含中心节点）
    num_nodes: int = 36
    # 时空立方体初始半窗口大小
    initial_radius: int = 4
    # 最大半窗口大小（防止无限扩展）
    max_radius: int = 20
    # 空间分区数（基于体对角线划分）
    num_regions: int = 6
    # Bresenham线段最大长度
    max_bresenham_len: int = 30


@dataclass
class AccelerationConfig:
    """加速策略配置"""
    # ---- 空间分块并行 ----
    tile_size: int = 128          # 瓦片大小（像素）
    num_workers: int = 8          # 并行进程数

    # ---- GPU加速 ----
    use_cuda: bool = True         # 是否使用CUDA
    gpu_batch_size: int = 4096    # GPU单次处理的线段数
    bresenham_lookup: bool = True # 是否启用Bresenham查找表

    # ---- 缓存复用 ----
    use_cache: bool = True        # 是否启用缓存
    cache_quantization: int = 2   # 缓存量化步长
    cache_max_size: int = 50000   # LRU缓存最大条目数

    # ---- 输出 ----
    output_format: str = "pkl"    # 输出格式
    save_per_tile: bool = True    # 按瓦片保存（减少内存峰值）


@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    accel: AccelerationConfig = field(default_factory=AccelerationConfig)

    # 输入输出路径
    input_path: str = ""
    output_dir: str = "./output_graphs"
    cache_dir: str = "./graph_cache"

    # 随机种子
    seed: int = 0

    def __post_init__(self):
        """初始化后自动创建目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_effective_temporal_range(self):
        """获取有效时间范围（去除缓冲区，自适应数据形状）"""
        T = self.data.data_shape[0]
        buf = min(self.data.temporal_buffer, T // 2)
        return buf, T - buf

    def get_effective_spatial_range(self):
        """获取有效空间范围（去除缓冲区，自适应数据形状）"""
        buf = min(self.data.buffer_size, min(self.data.data_shape[1], self.data.data_shape[2]) // 2)
        H, W = self.data.data_shape[1], self.data.data_shape[2]
        return buf, H - buf, buf, W - buf
