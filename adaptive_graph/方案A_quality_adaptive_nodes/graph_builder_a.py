"""
方案A图构建器 - 数据质量驱动的自适应节点数
=============================================
继承 v2 的 GraphBuilder，重写节点选择逻辑，实现自适应节点数。

核心改动：
  - build_single: 在选择节点前，先查询质量分数，动态决定节点数
  - _select_nodes_adaptive: 替代固定节点数的选择逻辑
  - 保留 v2 的边构建、缓存、Bresenham 查找表等基础设施

与 v2 的接口兼容性：
  - SubGraph 数据结构完全不变
  - 输出格式完全不变
  - 可与 v2 的 SpatialPartitioner、GraphCache 等模块无缝配合
"""

import numpy as np
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .config_a import ConfigA
from .adaptive_selector import AdaptiveNodeSelector

# 复用 v2 的基础设施
from ntl_graph_accel_v2.config import Config as V2Config
from ntl_graph_accel_v2.bresenham_lut import BresenhamLUT
from ntl_graph_accel_v2.graph_cache import GraphCache, GraphTemplate
from ntl_graph_accel_v2.graph_builder import SubGraph
from ntl_graph_accel_v2.jit_kernels import (
    _select_nodes_numba,
    _build_edges_numba,
    _count_region_valid_numba,
)

logger = logging.getLogger(__name__)


class GraphBuilderA:
    """
    方案A：数据质量驱动的自适应图构建器。

    与 v2 GraphBuilder 的区别：
    - 节点数不再固定为 36，而是根据局部数据质量动态调整
    - 高质量区域（有效像素密集、空间连续、时序稳定）→ 更多节点（最多 54）
    - 低质量区域（稀疏、断裂、波动大）→ 更少节点（最少 18）
    - 减少低质量区域的冗余计算，提高高质量区域的表示精度
    """

    def __init__(self, config: ConfigA, data: np.ndarray, valid_mask: np.ndarray):
        """
        Parameters
        ----------
        config : ConfigA
            方案A配置
        data : (T, H, W) float32
            全局数据
        valid_mask : (T, H, W) bool
            全局有效像素掩码
        """
        self.config = config
        self.data = data
        self.valid_mask = valid_mask
        self.T, self.H, self.W = data.shape

        # ---- 自适应节点数选择器 ----
        self.selector = AdaptiveNodeSelector(config.quality)

        # ---- Bresenham 查找表（复用 v2） ----
        self.lut: Optional[BresenhamLUT] = None
        if config.accel.bresenham_lookup:
            self.lut = BresenhamLUT(max_radius=config.graph.max_radius)
            self.lut.get_or_build(config.cache_dir)

        # ---- 缓存（复用 v2） ----
        self.cache: Optional[GraphCache] = None
        if config.accel.use_cache:
            self.cache = GraphCache(
                cache_dir=config.cache_dir,
                max_size=config.accel.cache_max_size,
                quantization_step=config.accel.cache_quantization
            )

        # ---- 预热 Numba JIT（在 lut 和 cache 初始化之后） ----
        if config.accel.use_numba:
            logger.info("[方案A] 预热 Numba JIT 编译...")
            self._warmup_jit()
            logger.info("[方案A] Numba JIT 编译完成")

        # ---- 统计信息 ----
        self._stats = {
            'total_queries': 0,
            'adaptive_counts': {},  # 节点数 → 出现次数
        }

    def _warmup_jit(self):
        """预热所有 Numba JIT 函数"""
        # 预热自适应选择器
        AdaptiveNodeSelector.warmup()

        # 预热 v2 的 JIT 内核
        tiny_valid = np.ones((5, 5, 5), dtype=np.bool_)
        tiny_data = np.random.rand(5, 5, 5).astype(np.float32)
        _select_nodes_numba(tiny_valid, tiny_data, 36, 6)

        tiny_offsets = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.int32)
        if self.lut is not None and self.lut.lut_array is not None:
            _build_edges_numba(
                tiny_offsets, tiny_valid,
                self.lut.lut_array, self.lut.lut_lengths, self.lut.max_radius
            )

    def precompute_quality(self):
        """
        预计算全局质量图。

        在批量构建之前调用，可显著提升性能。
        预计算后，get_num_nodes 查询为 O(1)。
        """
        logger.info("[方案A] 预计算全局质量指标图...")
        self.selector.precompute(self.valid_mask, self.data)
        logger.info("[方案A] 质量指标图预计算完成")

    def get_num_nodes(self, hc: int, wc: int) -> Tuple[int, float]:
        """
        查询指定空间位置的自适应节点数。

        Parameters
        ----------
        hc, wc : int
            空间位置坐标

        Returns
        -------
        num_nodes : int
            自适应节点数
        quality : float
            该位置的综合质量分数
        """
        n, q = self.selector.get_num_nodes(hc, wc, self.config.graph.num_nodes_base)
        self._stats['total_queries'] += 1
        self._stats['adaptive_counts'][n] = self._stats['adaptive_counts'].get(n, 0) + 1
        return n, q

    def build_single(self, tc: int, hc: int, wc: int) -> Optional[SubGraph]:
        """
        为单个位置构建子图（自适应节点数版本）。

        与 v2 的区别：
        1. 先查询该位置的质量分数，确定自适应节点数
        2. 用自适应节点数替代固定的 num_nodes 进行节点选择
        3. SubGraph 结构完全不变，仅 num_nodes 字段值不同

        Parameters
        ----------
        tc, hc, wc : int
            时空位置坐标

        Returns
        -------
        SubGraph 或 None
        """
        graph_cfg = self.config.graph

        # 尝试缓存命中
        if self.cache is not None:
            template = self.cache.get(tc, hc, wc)
            if template is not None:
                return self._build_from_template(tc, hc, wc, template)

        # ---- 方案A核心：查询自适应节点数 ----
        num_nodes, quality = self.get_num_nodes(hc, wc)

        # 自适应窗口扩展
        radius = graph_cfg.initial_radius
        cube, valid_cube = self._crop_cube(tc, hc, wc, radius)

        while True:
            counts = _count_region_valid_numba(valid_cube, graph_cfg.num_regions)
            q, rem = divmod(num_nodes, graph_cfg.num_regions)
            min_required = q + (1 if rem > 0 else 0)

            if counts.min() >= min_required:
                break

            radius += 1
            if radius > graph_cfg.max_radius:
                return None
            cube, valid_cube = self._crop_cube(tc, hc, wc, radius)

        # Numba JIT 节点选择（使用自适应节点数）
        offsets, features, regions, actual_n = _select_nodes_numba(
            valid_cube, cube, num_nodes, graph_cfg.num_regions
        )
        if actual_n < 2:
            return None

        # Numba JIT 边构建（与 v2 完全一致）
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

    def build_batch(self, positions: np.ndarray,
                    precompute_quality: bool = True) -> List[SubGraph]:
        """
        批量构建子图。

        Parameters
        ----------
        positions : (N, 3) int
            时空位置坐标数组
        precompute_quality : bool
            是否在批量构建前预计算质量图（默认 True，推荐开启）

        Returns
        -------
        List[SubGraph]
        """
        graphs = []
        total = len(positions)

        # 批量构建前预计算质量图
        if precompute_quality and not self.selector._precomputed:
            self.precompute_quality()

        for i, (tc, hc, wc) in enumerate(positions):
            graph = self.build_single(int(tc), int(hc), int(wc))
            if graph is not None:
                graphs.append(graph)
            if (i + 1) % 10000 == 0:
                hr = self.cache.get_hit_rate() if self.cache else 0
                logger.info(f"[方案A] 进度: {i+1}/{total}, "
                             f"已构建: {len(graphs)}, 缓存命中率: {hr:.1%}")

        # 打印统计信息
        self._print_stats(total, len(graphs))

        return graphs

    def _crop_cube(self, tc, hc, wc, radius):
        """裁剪时空子立方体（与 v2 完全一致）"""
        t0 = max(0, tc - radius)
        t1 = min(self.T, tc + radius + 1)
        h0 = max(0, hc - radius)
        h1 = min(self.H, hc + radius + 1)
        w0 = max(0, wc - radius)
        w1 = min(self.W, wc + radius + 1)
        return (self.data[t0:t1, h0:h1, w0:w1].copy(),
                self.valid_mask[t0:t1, h0:h1, w0:w1].copy())

    def _build_from_template(self, tc, hc, wc, template):
        """从缓存模板快速构建（与 v2 完全一致）"""
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

    def _print_stats(self, total_positions, total_graphs):
        """打印方案A的统计信息"""
        logger.info("=" * 60)
        logger.info("[方案A] 构建统计")
        logger.info(f"  总位置数: {total_positions}")
        logger.info(f"  成功构建: {total_graphs}")
        logger.info(f"  成功率:   {total_graphs / max(total_positions, 1):.1%}")
        logger.info(f"  质量查询: {self._stats['total_queries']}")

        if self._stats['adaptive_counts']:
            logger.info("  节点数分布:")
            for n in sorted(self._stats['adaptive_counts'].keys()):
                cnt = self._stats['adaptive_counts'][n]
                pct = cnt / max(self._stats['total_queries'], 1) * 100
                bar = '#' * int(pct / 2)
                logger.info(f"    N={n:3d}: {cnt:8d} ({pct:5.1f}%) {bar}")

        logger.info("=" * 60)

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'total_queries': self._stats['total_queries'],
            'adaptive_counts': dict(self._stats['adaptive_counts']),
            'cache_hit_rate': self.cache.get_hit_rate() if self.cache else 0.0,
        }
