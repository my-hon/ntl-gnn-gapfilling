"""
核心图构建模块（v2）
====================
调用 jit_kernels 中的 Numba JIT 函数，消除 Python 循环瓶颈。
保留缓存复用机制。
"""

import numpy as np
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .config import Config
from .bresenham_lut import BresenhamLUT
from .graph_cache import GraphCache, GraphTemplate
from .jit_kernels import (
    _select_nodes_numba,
    _build_edges_numba,
    _count_region_valid_numba,
)

logger = logging.getLogger(__name__)


@dataclass
class SubGraph:
    """子图数据结构（与 v1 兼容）"""
    center_pos: np.ndarray
    node_features: np.ndarray
    edge_index_src: np.ndarray
    edge_index_dst: np.ndarray
    edge_attrs: np.ndarray
    center_value: float
    num_nodes: int

    def to_dict(self):
        return {
            'center_pos': self.center_pos,
            'node_features': self.node_features,
            'edge_index': np.stack([self.edge_index_src, self.edge_index_dst], axis=0),
            'edge_attrs': self.edge_attrs,
            'center_value': self.center_value,
            'num_nodes': self.num_nodes
        }


class GraphBuilder:
    """
    时空图构建器（v2 - Numba JIT 加速版）。

    与 v1 的区别：
    - _select_nodes: 调用 Numba JIT 内核，消除 Python 循环
    - _build_edges:  调用 Numba JIT 内核，用数组查找替代 dict
    - _count_region_valid: 调用 Numba JIT 内核
    - 移除了无效的 GPU Bresenham 加速器
    """

    def __init__(self, config: Config, data: np.ndarray, valid_mask: np.ndarray):
        self.config = config
        self.data = data
        self.valid_mask = valid_mask
        self.T, self.H, self.W = data.shape

        # 查找表
        self.lut: Optional[BresenhamLUT] = None
        if config.accel.bresenham_lookup:
            self.lut = BresenhamLUT(max_radius=config.graph.max_radius)
            self.lut.get_or_build(config.cache_dir)

        # 缓存
        self.cache: Optional[GraphCache] = None
        if config.accel.use_cache:
            self.cache = GraphCache(
                cache_dir=config.cache_dir,
                max_size=config.accel.cache_max_size,
                quantization_step=config.accel.cache_quantization
            )

        # 预热 Numba JIT（首次调用会触发编译）
        if config.accel.use_numba:
            logger.info("预热 Numba JIT 编译...")
            self._warmup_jit()
            logger.info("Numba JIT 编译完成")

    def _warmup_jit(self):
        """用小数据触发 Numba JIT 编译"""
        tiny_valid = np.ones((5, 5, 5), dtype=np.bool_)
        tiny_data = np.random.rand(5, 5, 5).astype(np.float32)
        _select_nodes_numba(tiny_valid, tiny_data, 4, 6)

        tiny_offsets = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.int32)
        if self.lut is not None and self.lut.lut_array is not None:
            _build_edges_numba(
                tiny_offsets, tiny_valid,
                self.lut.lut_array, self.lut.lut_lengths, self.lut.max_radius
            )

    def build_single(self, tc: int, hc: int, wc: int) -> Optional[SubGraph]:
        """为单个位置构建子图"""
        graph_cfg = self.config.graph

        # 尝试缓存
        if self.cache is not None:
            template = self.cache.get(tc, hc, wc)
            if template is not None:
                return self._build_from_template(tc, hc, wc, template)

        # 自适应窗口扩展
        radius = graph_cfg.initial_radius
        cube, valid_cube = self._crop_cube(tc, hc, wc, radius)

        while True:
            counts = _count_region_valid_numba(valid_cube, graph_cfg.num_regions)
            q, rem = divmod(graph_cfg.num_nodes, graph_cfg.num_regions)
            min_required = q + (1 if rem > 0 else 0)

            if counts.min() >= min_required:
                break

            radius += 1
            if radius > graph_cfg.max_radius:
                return None
            cube, valid_cube = self._crop_cube(tc, hc, wc, radius)

        # Numba JIT 节点选择
        offsets, features, regions, actual_n = _select_nodes_numba(
            valid_cube, cube, graph_cfg.num_nodes, graph_cfg.num_regions
        )
        if actual_n < 2:
            return None

        # Numba JIT 边构建
        if self.lut is not None and self.lut.lut_array is not None:
            edge_src, edge_dst, edge_attrs, num_edges = _build_edges_numba(
                offsets, valid_cube,
                self.lut.lut_array, self.lut.lut_lengths, self.lut.max_radius
            )
        else:
            edge_src, edge_dst, edge_attrs, num_edges = _build_edges_numba(
                offsets, valid_cube,
                np.zeros((1,1,1,1,3), dtype=np.int16),
                np.zeros((1,1,1), dtype=np.int16),
                0
            )

        # 归一化
        features = features / self.config.data.feature_scale
        edge_attrs = edge_attrs / self.config.data.edge_scale

        center_value = float(self.data[tc, hc, wc])

        subgraph = SubGraph(
            center_pos=np.array([tc, hc, wc], dtype=np.int32),
            node_features=features.astype(np.float32),
            edge_index_src=edge_src.astype(np.int64),
            edge_index_dst=edge_dst.astype(np.int64),
            edge_attrs=edge_attrs.astype(np.float32),
            center_value=center_value,
            num_nodes=actual_n
        )

        # 存入缓存
        if self.cache is not None:
            template = GraphTemplate(
                node_offsets=offsets.copy(),
                edge_src=edge_src.copy(),
                edge_dst=edge_dst.copy(),
                edge_attrs=edge_attrs.copy(),
                self_loop_indices=np.arange(actual_n, dtype=np.int64),
                region_counts=np.array([np.sum(regions == i) for i in range(graph_cfg.num_regions)]),
                radius=radius
            )
            self.cache.put(tc, hc, wc, template)

        return subgraph

    def build_batch(self, positions: np.ndarray) -> List[SubGraph]:
        """批量构建子图"""
        graphs = []
        total = len(positions)
        for i, (tc, hc, wc) in enumerate(positions):
            graph = self.build_single(int(tc), int(hc), int(wc))
            if graph is not None:
                graphs.append(graph)
            if (i + 1) % 10000 == 0:
                hr = self.cache.get_hit_rate() if self.cache else 0
                logger.info(f"进度: {i+1}/{total}, 已构建: {len(graphs)}, 缓存命中率: {hr:.1%}")
        return graphs

    def _crop_cube(self, tc, hc, wc, radius):
        """裁剪时空子立方体"""
        t0 = max(0, tc - radius)
        t1 = min(self.T, tc + radius + 1)
        h0 = max(0, hc - radius)
        h1 = min(self.H, hc + radius + 1)
        w0 = max(0, wc - radius)
        w1 = min(self.W, wc + radius + 1)
        return (self.data[t0:t1, h0:h1, w0:w1].copy(),
                self.valid_mask[t0:t1, h0:h1, w0:w1].copy())

    def _build_from_template(self, tc, hc, wc, template):
        """从缓存模板快速构建"""
        radius = template.radius
        cube, _ = self._crop_cube(tc, hc, wc, radius)
        ct = cube.shape[0] // 2
        ch = cube.shape[1] // 2
        cw = cube.shape[2] // 2

        features = np.array([
            cube[off[0] + ct, off[1] + ch, off[2] + cw]
            for off in template.node_offsets
        ], dtype=np.float32) / self.config.data.feature_scale

        return SubGraph(
            center_pos=np.array([tc, hc, wc], dtype=np.int32),
            node_features=features,
            edge_index_src=template.edge_src.copy(),
            edge_index_dst=template.edge_dst.copy(),
            edge_attrs=template.edge_attrs / self.config.data.edge_scale,
            center_value=float(self.data[tc, hc, wc]),
            num_nodes=len(template.node_offsets)
        )
