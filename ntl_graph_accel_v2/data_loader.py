"""
数据加载与预处理模块（v2）
==========================
与 v1 逻辑一致，移除了不必要的质量图预计算（v2 中未使用）。
"""

import numpy as np
import logging
from typing import Optional
from .config import Config

logger = logging.getLogger(__name__)


class NTLDataLoader:
    """夜间灯光数据加载器"""

    def __init__(self, config: Config):
        self.config = config
        self.data = None
        self.valid_mask = None

    def load(self, data: Optional[np.ndarray] = None, path: Optional[str] = None):
        if data is not None:
            self.data = data.astype(np.float32)
        elif path is not None:
            self.data = np.load(path).astype(np.float32)
        else:
            raise ValueError("必须提供 data 或 path 参数")

        assert self.data.ndim == 3, f"数据应为3维(T,H,W)，当前维度: {self.data.ndim}"
        T, H, W = self.data.shape
        logger.info(f"数据加载完成: 形状=({T}, {H}, {W}), dtype={self.data.dtype}")
        self.config.data.data_shape = (T, H, W)
        self._build_valid_mask()
        return self

    def _build_valid_mask(self):
        self.valid_mask = ~np.isnan(self.data)
        total = self.valid_mask.size
        valid = self.valid_mask.sum()
        logger.info(f"有效像素: {valid}/{total} ({100*valid/total:.1f}%)")

    def get_missing_positions(self) -> np.ndarray:
        t_start, t_end = self.config.get_effective_temporal_range()
        h_start, h_end, w_start, w_end = self.config.get_effective_spatial_range()
        region = self.data[t_start:t_end, h_start:h_end, w_start:w_end]
        nan_mask = np.isnan(region)
        ts, hs, ws = np.where(nan_mask)
        positions = np.stack([ts + t_start, hs + h_start, ws + w_start], axis=1)
        logger.info(f"缺失位置总数: {len(positions)}")
        return positions

    def get_all_valid_positions(self) -> np.ndarray:
        t_start, t_end = self.config.get_effective_temporal_range()
        h_start, h_end, w_start, w_end = self.config.get_effective_spatial_range()
        region = self.data[t_start:t_end, h_start:h_end, w_start:w_end]
        valid = ~np.isnan(region)
        ts, hs, ws = np.where(valid)
        positions = np.stack([ts + t_start, hs + h_start, ws + w_start], axis=1)
        logger.info(f"有效位置总数: {len(positions)}")
        return positions
