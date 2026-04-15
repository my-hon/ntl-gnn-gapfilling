"""
数据加载与预处理模块
====================
负责加载NumPy格式的NTL数据，识别有效/缺失位置，并提供数据访问接口。
"""

import numpy as np
import logging
from typing import Tuple, Optional
from .config import Config

logger = logging.getLogger(__name__)


class NTLDataLoader:
    """夜间灯光数据加载器"""

    def __init__(self, config: Config):
        self.config = config
        self.data = None          # (T, H, W) 完整数据
        self.valid_mask = None    # (T, H, W) 有效像素掩码
        self.quality_map = None   # (T, H, W) 局部数据质量指标

    def load(self, data: Optional[np.ndarray] = None, path: Optional[str] = None):
        """
        加载数据。

        Parameters
        ----------
        data : np.ndarray, optional
            直接传入 (T, H, W) 数组
        path : str, optional
            从.npy文件加载
        """
        if data is not None:
            self.data = data.astype(np.float32)
        elif path is not None:
            self.data = np.load(path).astype(np.float32)
        else:
            raise ValueError("必须提供 data 或 path 参数")

        assert self.data.ndim == 3, f"数据应为3维(T,H,W)，当前维度: {self.data.ndim}"
        T, H, W = self.data.shape
        logger.info(f"数据加载完成: 形状=({T}, {H}, {W}), "
                     f"dtype={self.data.dtype}")

        # 更新配置中的形状
        self.config.data.data_shape = (T, H, W)

        # 构建有效掩码
        self._build_valid_mask()

        # 预计算局部数据质量图
        self._precompute_quality_map()

        return self

    def _build_valid_mask(self):
        """构建有效像素掩码（非NaN且非零的为有效值）"""
        self.valid_mask = ~np.isnan(self.data)
        total = self.valid_mask.size
        valid = self.valid_mask.sum()
        logger.info(f"有效像素: {valid}/{total} ({100*valid/total:.1f}%)")

    def _precompute_quality_map(self):
        """
        预计算局部数据质量图。
        使用滑动窗口计算每个位置邻域内的有效像素密度，
        用于后续自适应节点数调整。
        """
        T, H, W = self.data.shape
        r = 3  # 质量计算窗口半径
        self.quality_map = np.zeros((T, H, W), dtype=np.float32)

        # 对每个时间切片计算空间质量
        for t in range(T):
            slice_valid = self.valid_mask[t].astype(np.float32)
            # 使用积分图加速密度计算
            integral = np.zeros((H + 1, W + 1), dtype=np.float32)
            integral[1:, 1:] = np.cumsum(
                np.cumsum(slice_valid, axis=0), axis=1
            )
            # 滑动窗口求和
            for h in range(H):
                for w in range(W):
                    h1, h2 = max(0, h - r), min(H, h + r + 1)
                    w1, w2 = max(0, w - r), min(W, w + r + 1)
                    count = (integral[h2, w2] - integral[h1, w2]
                             - integral[h2, w1] + integral[h1, w1])
                    area = (h2 - h1) * (w2 - w1)
                    self.quality_map[t, h, w] = count / max(area, 1)

        logger.info("局部数据质量图计算完成")

    def get_cube(self, tc: int, hc: int, wc: int, radius: int) -> np.ndarray:
        """
        获取以(tc, hc, wc)为中心、半径为radius的时空立方体。

        Parameters
        ----------
        tc, hc, wc : int
            中心坐标
        radius : int
            半窗口大小

        Returns
        -------
        np.ndarray
            形状为 (2*radius+1, 2*radius+1, 2*radius+1) 的立方体
        """
        T, H, W = self.data.shape
        t1 = max(0, tc - radius)
        t2 = min(T, tc + radius + 1)
        h1 = max(0, hc - radius)
        h2 = min(H, hc + radius + 1)
        w1 = max(0, wc - radius)
        w2 = min(W, wc + radius + 1)
        return self.data[t1:t2, h1:h2, w1:w2]

    def get_valid_cube(self, tc: int, hc: int, wc: int, radius: int) -> np.ndarray:
        """获取有效掩码的时空立方体"""
        T, H, W = self.valid_mask.shape
        t1 = max(0, tc - radius)
        t2 = min(T, tc + radius + 1)
        h1 = max(0, hc - radius)
        h2 = min(H, hc + radius + 1)
        w1 = max(0, wc - radius)
        w2 = min(W, wc + radius + 1)
        return self.valid_mask[t1:t2, h1:h2, w1:w2]

    def get_missing_positions(self) -> np.ndarray:
        """
        获取所有需要填补的缺失位置（有效区域内的NaN位置）。

        Returns
        -------
        np.ndarray
            形状为 (N, 3) 的数组，每行为 (t, h, w)
        """
        t_start, t_end = self.config.get_effective_temporal_range()
        h_start, h_end, w_start, w_end = self.config.get_effective_spatial_range()

        # 在有效区域内找NaN位置
        region = self.data[t_start:t_end, h_start:h_end, w_start:w_end]
        nan_mask = np.isnan(region)

        # 获取NaN位置的索引（相对于有效区域）
        ts, hs, ws = np.where(nan_mask)
        # 转换为全局坐标
        positions = np.stack([
            ts + t_start,
            hs + h_start,
            ws + w_start
        ], axis=1)

        logger.info(f"缺失位置总数: {len(positions)}")
        return positions

    def get_all_valid_positions(self) -> np.ndarray:
        """
        获取有效区域内所有有效像素位置（用于训练数据构建）。

        Returns
        -------
        np.ndarray
            形状为 (N, 3) 的数组
        """
        t_start, t_end = self.config.get_effective_temporal_range()
        h_start, h_end, w_start, w_end = self.config.get_effective_spatial_range()

        region = self.data[t_start:t_end, h_start:h_end, w_start:w_end]
        valid = ~np.isnan(region)

        ts, hs, ws = np.where(valid)
        positions = np.stack([
            ts + t_start,
            hs + h_start,
            ws + w_start
        ], axis=1)

        logger.info(f"有效位置总数: {len(positions)}")
        return positions

    def normalize_node_features(self, values: np.ndarray) -> np.ndarray:
        """归一化节点特征值"""
        return values / self.config.data.feature_scale

    def normalize_edge_attrs(self, offsets: np.ndarray) -> np.ndarray:
        """归一化边属性（3D偏移量）"""
        return offsets / self.config.data.edge_scale
