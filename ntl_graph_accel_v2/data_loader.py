"""
数据加载与预处理模块（v2）
==========================
与参考实现 build_dataset.py 的 read_data 精确对齐。

关键改动：
  1. 支持质量标志文件加载（quality_path）
  2. 预处理流程：65535→NaN, quality>1→NaN, /10.0, /100.0
  3. 自然断点采样方法
  4. 逐天 NaN 位置提取方法
"""

import numpy as np
import logging
from typing import Optional, List, Tuple
from .config import Config

logger = logging.getLogger(__name__)


class NTLDataLoader:
    """夜间灯光数据加载器"""

    def __init__(self, config: Config):
        self.config = config
        self.data = None          # 预处理后的归一化数据 (float32)
        self.valid_mask = None    # 有效像素掩码

    def load(self, data: Optional[np.ndarray] = None, path: Optional[str] = None,
             quality_data: Optional[np.ndarray] = None, quality_path: Optional[str] = None):
        """
        加载并预处理数据。

        参数:
            data: 原始 uint16 数据数组 (T, H, W)
            path: 原始数据文件路径 (.npy)
            quality_data: 质量标志数据数组 (T, H, W)
            quality_path: 质量标志文件路径 (.npy)
        """
        # 加载原始数据
        if data is not None:
            raw_data = data
        elif path is not None:
            raw_data = np.load(path)
        else:
            raise ValueError("必须提供 data 或 path 参数")

        assert raw_data.ndim == 3, f"数据应为3维(T,H,W)，当前维度: {raw_data.ndim}"
        T, H, W = raw_data.shape
        logger.info(f"原始数据加载完成: 形状=({T}, {H}, {W}), dtype={raw_data.dtype}")

        # 加载质量标志
        if quality_data is not None:
            quality = quality_data
        elif quality_path is not None:
            quality = np.load(quality_path)
        elif self.config.data.quality_path:
            quality = np.load(self.config.data.quality_path)
        else:
            quality = None

        if quality is not None:
            logger.info(f"质量标志加载完成: 形状={quality.shape}, dtype={quality.dtype}")

        # 预处理：精确复制参考实现 read_data 的逻辑
        self.data = self._preprocess(raw_data, quality)

        self.config.data.data_shape = (T, H, W)
        self._build_valid_mask()
        return self

    def _preprocess(self, raw_data: np.ndarray, quality: Optional[np.ndarray] = None) -> np.ndarray:
        """
        预处理流程，精确复制参考实现：
        1. uint16 65535 → NaN
        2. quality > 1 → NaN
        3. /10.0 恢复真实辐射值
        4. /100.0 归一化
        """
        img_data = raw_data.astype(np.float64)

        # 65535 → NaN
        img_data = np.where(img_data == 65535, np.nan, img_data)

        # quality > 1 → NaN
        if quality is not None:
            img_data = np.where(quality > 1, np.nan, img_data)

        # /10.0 恢复真实辐射值
        img_data = img_data / 10.0

        logger.info(
            f"预处理完成 (/10.0): shape={img_data.shape}, "
            f"max={np.nanmax(img_data):.4f}, min={np.nanmin(img_data):.4f}, "
            f"99.9%={np.nanpercentile(img_data, q=99.9):.4f}"
        )

        # /100.0 归一化
        img_data_norm = img_data / 100.0

        logger.info(
            f"归一化完成 (/100.0): max={np.nanmax(img_data_norm):.6f}, "
            f"min={np.nanmin(img_data_norm):.6f}"
        )

        return img_data_norm.astype(np.float32)

    def _build_valid_mask(self):
        self.valid_mask = ~np.isnan(self.data)
        total = self.valid_mask.size
        valid = self.valid_mask.sum()
        logger.info(f"有效像素: {valid}/{total} ({100*valid/total:.1f}%)")

    def get_effective_region(self) -> np.ndarray:
        """
        获取有效区域内的数据。
        有效区域: data[edge_time:-edge_time, edge_height:-edge_height, edge_width:-edge_width]
        """
        t_start, t_end = self.config.get_effective_temporal_range()
        h_start, h_end, w_start, w_end = self.config.get_effective_spatial_range()
        return self.data[t_start:t_end, h_start:h_end, w_start:w_end]

    def get_missing_positions(self) -> np.ndarray:
        """
        获取有效区域内的 NaN 位置（全局坐标）。
        """
        t_start, t_end = self.config.get_effective_temporal_range()
        h_start, h_end, w_start, w_end = self.config.get_effective_spatial_range()
        region = self.data[t_start:t_end, h_start:h_end, w_start:w_end]
        nan_mask = np.isnan(region)
        ts, hs, ws = np.where(nan_mask)
        positions = np.stack([ts + t_start, hs + h_start, ws + w_start], axis=1)
        logger.info(f"缺失位置总数: {len(positions)}")
        return positions

    def get_all_valid_positions(self) -> np.ndarray:
        """
        获取有效区域内的非 NaN 位置（全局坐标）。
        """
        t_start, t_end = self.config.get_effective_temporal_range()
        h_start, h_end, w_start, w_end = self.config.get_effective_spatial_range()
        region = self.data[t_start:t_end, h_start:h_end, w_start:w_end]
        valid = ~np.isnan(region)
        ts, hs, ws = np.where(valid)
        positions = np.stack([ts + t_start, hs + h_start, ws + w_start], axis=1)
        logger.info(f"有效位置总数: {len(positions)}")
        return positions

    def get_natural_breaks_samples(self) -> List[Tuple[int, np.ndarray]]:
        """
        自然断点采样。
        精确复制参考实现的 train_data_generate 采样逻辑。

        返回:
            List of (class_index, positions_array) 元组。
            positions_array 的坐标为全局坐标。
        """
        natural_breaks = self.config.graph.natural_breaks
        sample_pc = self.config.graph.sample_per_class

        cut_data = self.get_effective_region()

        t_start, t_end = self.config.get_effective_temporal_range()
        h_start, h_end, w_start, w_end = self.config.get_effective_spatial_range()
        offset = np.array([[t_start, h_start, w_start]])

        results = []
        for i in range(len(natural_breaks) - 1):
            indices = np.array(np.where(
                (cut_data > natural_breaks[i]) & (cut_data < natural_breaks[i + 1])
            )).T + offset

            if len(indices) == 0:
                logger.info(f"类别 {i+1}: [{natural_breaks[i]}, {natural_breaks[i+1]}] - 无有效像素")
                results.append((i, np.empty((0, 3), dtype=np.int64)))
                continue

            random_choice = np.random.choice(
                np.arange(len(indices)),
                size=sample_pc,
                replace=len(indices) < sample_pc,
            )
            selected_indices = indices[random_choice]

            logger.info(
                f"类别 {i+1}: [{natural_breaks[i]}, {natural_breaks[i+1]}], "
                f"有效像素: {len(indices)}, 采样: {len(selected_indices)}"
            )
            results.append((i, selected_indices))

        return results

    def get_nan_positions_by_day(self) -> List[Tuple[int, np.ndarray]]:
        """
        逐天提取 NaN 位置。
        精确复制参考实现的 predict_data_generate 逻辑。

        返回:
            List of (day_number, positions_array) 元组。
            positions_array 的坐标为全局坐标。
        """
        cut_data = self.get_effective_region()

        t_start, t_end = self.config.get_effective_temporal_range()
        h_start, h_end, w_start, w_end = self.config.get_effective_spatial_range()
        offset = np.array([[t_start, h_start, w_start]])

        indices_nan = np.array(np.where(np.isnan(cut_data))).T

        results = []
        total_days = self.data.shape[0] - 2 * self.config.data.edge_time
        for day_num in range(total_days):
            day_mask = indices_nan[:, 0] == day_num
            indices_nan_day = indices_nan[day_mask] + offset

            results.append((day_num, indices_nan_day))

        logger.info(f"逐天 NaN 提取完成: 共 {total_days} 天, "
                     f"总 NaN 位置: {len(indices_nan)}")
        return results
