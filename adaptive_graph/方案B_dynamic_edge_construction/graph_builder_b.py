"""
方案B图构建器模块
=================
空间异质性感知的动态连边图构建器。

与 v2 GraphBuilder 的区别：
  1. 节点选择：复用 v2 的 _select_nodes_numba（逻辑不变）
  2. 边构建：使用方案B的 _build_edges_dynamic_with_heterogeneity 替代 v2 的 _build_edges_numba
  3. 异质性分析：在边构建前自动计算局部异质性指数
  4. 动态权重：根据异质性指数动态调整边属性和连接策略
  5. SubGraph 接口：完全兼容 v2 的 SubGraph 数据结构

灵感来源：
  - Graph WaveNet (arXiv:1906.00121): 自适应邻接矩阵
  - DRTR (arXiv:2406.17281): 距离感知拓扑精炼
"""

import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntl_graph_accel_v2.config import Config
from ntl_graph_accel_v2.bresenham_lut import BresenhamLUT
from ntl_graph_accel_v2.graph_cache import GraphCache, GraphTemplate
from ntl_graph_accel_v2.jit_kernels import (
    _select_nodes_numba,
    _count_region_valid_numba,
)

from .config_b import ConfigB
from .heterogeneity_analyzer import HeterogeneityAnalyzer
from .dynamic_edge_builder import DynamicEdgeBuilder

logger = logging.getLogger(__name__)


@dataclass
class SubGraph:
    """子图数据结构（与 v2 完全兼容）

    方案B额外记录了异质性信息，但保持基础接口不变。
    """
    center_pos: np.ndarray
    node_features: np.ndarray
    edge_index_src: np.ndarray
    edge_index_dst: np.ndarray
    edge_attrs: np.ndarray
    center_value: float
    num_nodes: int
    # 方案B扩展字段
    heterogeneity_index: float = -1.0     # 异质性指数 H
    w_spatial: float = 0.5                # 空间权重
    w_temporal: float = 0.5               # 时序权重

    def to_dict(self):
        """转换为字典（与 v2 兼容，忽略扩展字段）"""
        return {
            'center_pos': self.center_pos,
            'node_features': self.node_features,
            'edge_index': np.stack([self.edge_index_src, self.edge_index_dst], axis=0),
            'edge_attrs': self.edge_attrs,
            'center_value': self.center_value,
            'num_nodes': self.num_nodes
        }

    def to_dict_full(self):
        """转换为完整字典（包含方案B扩展字段）"""
        d = self.to_dict()
        d['heterogeneity_index'] = self.heterogeneity_index
        d['w_spatial'] = self.w_spatial
        d['w_temporal'] = self.w_temporal
        return d


class GraphBuilderB:
    """
    方案B：空间异质性感知的动态连边图构建器。

    工作流程：
      1. 对每个中心位置，裁剪时空子立方体
      2. 自适应窗口扩展（复用 v2 逻辑）
      3. Numba JIT 节点选择（复用 v2 的 _select_nodes_numba）
      4. 计算局部异质性指数 H = std(NTL) / mean(NTL)
      5. 根据异质性计算动态时空权重 w_spatial, w_temporal
      6. 使用动态权重构建边（方案B核心改进）
         - 高异质性：放松 Bresenham 截断，增加空间近邻连接
         - 低异质性：扩展时序连接窗口

    与 v2 GraphBuilder 的接口差异：
      - 构造函数接受 ConfigB（而非 Config）
      - 边构建使用 DynamicEdgeBuilder（而非 v2 的 _build_edges_numba）
      - SubGraph 包含异质性扩展字段（但基础接口兼容）
    """

    def __init__(self, config: ConfigB, data: np.ndarray, valid_mask: np.ndarray):
        """
        Parameters
        ----------
        config : ConfigB
            方案B配置（包含异质性参数）
        data : (T, H, W) float32
            NTL 数据
        valid_mask : (T, H, W) bool
            有效像素掩码
        """
        self.config = config
        self.data = data
        self.valid_mask = valid_mask
        self.T, self.H, self.W = data.shape

        het_cfg = config.heterogeneity

        # Bresenham 查找表（复用 v2）
        self.lut: Optional[BresenhamLUT] = None
        if config.accel.bresenham_lookup:
            self.lut = BresenhamLUT(max_radius=config.graph.max_radius)
            self.lut.get_or_build(config.cache_dir)

        # 缓存（复用 v2）
        self.cache: Optional[GraphCache] = None
        if config.accel.use_cache:
            self.cache = GraphCache(
                cache_dir=config.cache_dir,
                max_size=config.accel.cache_max_size,
                quantization_step=config.accel.cache_quantization
            )

        # 异质性分析器
        self.heterogeneity_analyzer = HeterogeneityAnalyzer(
            cube_radius=het_cfg.heterogeneity_cube_radius,
            h_threshold=het_cfg.h_threshold,
            h_scale=het_cfg.h_scale,
            min_valid=het_cfg.min_valid_for_heterogeneity,
        )

        # 动态边构建器
        self.dynamic_edge_builder = DynamicEdgeBuilder(
            h_threshold=het_cfg.h_threshold,
            h_scale=het_cfg.h_scale,
            spatial_boost=het_cfg.high_het_spatial_boost,
            temporal_extend=het_cfg.low_het_temporal_extend,
            min_valid=het_cfg.min_valid_for_heterogeneity,
        )

        # 预热 Numba JIT
        if config.accel.use_numba:
            logger.info("[方案B] 预热 Numba JIT 编译...")
            self._warmup_jit()
            logger.info("[方案B] Numba JIT 编译完成")

    def _warmup_jit(self):
        """用小数据触发所有 Numba JIT 编译"""
        tiny_valid = np.ones((5, 5, 5), dtype=np.bool_)
        tiny_data = np.random.rand(5, 5, 5).astype(np.float32)

        # 预热 v2 的节点选择内核
        _select_nodes_numba(tiny_valid, tiny_data, 4, 6)

        # 预热方案B的异质性分析内核
        from .heterogeneity_analyzer import _compute_heterogeneity_for_cube
        _compute_heterogeneity_for_cube(
            tiny_data, tiny_valid,
            self.config.heterogeneity.min_valid_for_heterogeneity,
            self.config.heterogeneity.h_threshold,
            self.config.heterogeneity.h_scale
        )

        # 预热方案B的动态边构建内核
        tiny_offsets = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.int32)
        if self.lut is not None and self.lut.lut_array is not None:
            from .dynamic_edge_builder import _build_edges_dynamic_numba
            _build_edges_dynamic_numba(
                tiny_offsets, tiny_valid,
                self.lut.lut_array, self.lut.lut_lengths, self.lut.max_radius,
                0.5, 0.5,  # w_spatial, w_temporal
                1.5, 3     # spatial_boost, temporal_extend
            )

    def build_single(self, tc: int, hc: int, wc: int) -> Optional[SubGraph]:
        """
        为单个位置构建子图（方案B核心方法）。

        Parameters
        ----------
        tc, hc, wc : int
            中心位置的时间、高度、宽度坐标

        Returns
        -------
        SubGraph 或 None
            构建成功返回 SubGraph，失败返回 None
        """
        graph_cfg = self.config.graph
        het_cfg = self.config.heterogeneity

        # 尝试缓存（方案B暂不缓存，因为异质性可能因数据不同而变化）
        # 但如果缓存命中且结构相同，可以复用拓扑
        # TODO: 实现异质性感知的缓存策略

        # 自适应窗口扩展（与 v2 逻辑一致）
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

        # Numba JIT 节点选择（复用 v2）
        offsets, features, regions, actual_n = _select_nodes_numba(
            valid_cube, cube, graph_cfg.num_nodes, graph_cfg.num_regions
        )
        if actual_n < 2:
            return None

        # ---- 方案B核心：异质性感知的动态边构建 ----
        if self.lut is not None and self.lut.lut_array is not None:
            # 使用合并的异质性+边构建内核（单次 Numba 调用）
            from .dynamic_edge_builder import _build_edges_dynamic_with_heterogeneity

            edge_src, edge_dst, edge_attrs, num_edges, H_val, ws, wt = \
                _build_edges_dynamic_with_heterogeneity(
                    offsets, valid_cube, cube,
                    self.lut.lut_array, self.lut.lut_lengths, self.lut.max_radius,
                    het_cfg.h_threshold,
                    het_cfg.h_scale,
                    het_cfg.high_het_spatial_boost,
                    het_cfg.low_het_temporal_extend,
                    het_cfg.min_valid_for_heterogeneity,
                )
        else:
            # 无 Bresenham 查找表时的回退
            from .dynamic_edge_builder import _build_edges_dynamic_with_heterogeneity

            edge_src, edge_dst, edge_attrs, num_edges, H_val, ws, wt = \
                _build_edges_dynamic_with_heterogeneity(
                    offsets, valid_cube, cube,
                    np.zeros((1, 1, 1, 1, 3), dtype=np.int16),
                    np.zeros((1, 1, 1), dtype=np.int16),
                    0,
                    het_cfg.h_threshold,
                    het_cfg.h_scale,
                    het_cfg.high_het_spatial_boost,
                    het_cfg.low_het_temporal_extend,
                    het_cfg.min_valid_for_heterogeneity,
                )

        # 归一化
        features = features / self.config.data.feature_scale
        edge_attrs = edge_attrs / self.config.data.edge_scale

        center_value = float(self.data[tc, hc, wc])

        # 构建方案B的 SubGraph（包含异质性信息）
        subgraph = SubGraph(
            center_pos=np.array([tc, hc, wc], dtype=np.int32),
            node_features=features.astype(np.float32),
            edge_index_src=edge_src.astype(np.int64),
            edge_index_dst=edge_dst.astype(np.int64),
            edge_attrs=edge_attrs.astype(np.float32),
            center_value=center_value,
            num_nodes=actual_n,
            heterogeneity_index=H_val,
            w_spatial=ws,
            w_temporal=wt,
        )

        # 存入缓存（使用 v2 的 GraphTemplate，忽略异质性信息）
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
        """
        批量构建子图。

        Parameters
        ----------
        positions : (N, 3) int32
            中心位置数组

        Returns
        -------
        List[SubGraph]
        """
        graphs = []
        total = len(positions)
        het_stats = {'high': 0, 'low': 0, 'invalid': 0}

        for i, (tc, hc, wc) in enumerate(positions):
            graph = self.build_single(int(tc), int(hc), int(wc))
            if graph is not None:
                graphs.append(graph)
                # 统计异质性分布
                if graph.heterogeneity_index >= self.config.heterogeneity.h_threshold:
                    het_stats['high'] += 1
                elif graph.heterogeneity_index >= 0:
                    het_stats['low'] += 1
                else:
                    het_stats['invalid'] += 1

            if (i + 1) % 10000 == 0:
                hr = self.cache.get_hit_rate() if self.cache else 0
                logger.info(
                    f"[方案B] 进度: {i+1}/{total}, 已构建: {len(graphs)}, "
                    f"缓存命中率: {hr:.1%}, "
                    f"高异质性: {het_stats['high']}, "
                    f"低异质性: {het_stats['low']}, "
                    f"无效: {het_stats['invalid']}"
                )

        logger.info(
            f"[方案B] 批量构建完成: 总位置={total}, 成功={len(graphs)}, "
            f"成功率={len(graphs)/max(total,1):.1%}, "
            f"高异质性区域: {het_stats['high']}, "
            f"低异质性区域: {het_stats['low']}, "
            f"异质性无效: {het_stats['invalid']}"
        )
        return graphs

    def build_batch_with_analysis(
        self, positions: np.ndarray
    ) -> Tuple[List[SubGraph], Dict]:
        """
        批量构建子图并返回异质性分析报告。

        Parameters
        ----------
        positions : (N, 3) int32

        Returns
        -------
        graphs : List[SubGraph]
        analysis : Dict
            异质性分析报告，包含：
            - heterogeneity_distribution: 异质性指数分布统计
            - weight_distribution: 时空权重分布统计
            - edge_count_stats: 边数统计
        """
        graphs = self.build_batch(positions)

        # 收集统计信息
        h_indices = [g.heterogeneity_index for g in graphs if g.heterogeneity_index >= 0]
        w_spatials = [g.w_spatial for g in graphs]
        w_temporals = [g.w_temporal for g in graphs]
        edge_counts = [
            len(g.edge_index_src) for g in graphs
        ]

        analysis = {
            'heterogeneity_distribution': {
                'mean': float(np.mean(h_indices)) if h_indices else -1.0,
                'std': float(np.std(h_indices)) if h_indices else -1.0,
                'min': float(np.min(h_indices)) if h_indices else -1.0,
                'max': float(np.max(h_indices)) if h_indices else -1.0,
                'median': float(np.median(h_indices)) if h_indices else -1.0,
                'num_valid': len(h_indices),
                'num_invalid': len(graphs) - len(h_indices),
            },
            'weight_distribution': {
                'w_spatial_mean': float(np.mean(w_spatials)) if w_spatials else 0.0,
                'w_spatial_std': float(np.std(w_spatials)) if w_spatials else 0.0,
                'w_temporal_mean': float(np.mean(w_temporals)) if w_temporals else 0.0,
                'w_temporal_std': float(np.std(w_temporals)) if w_temporals else 0.0,
            },
            'edge_count_stats': {
                'mean': float(np.mean(edge_counts)) if edge_counts else 0.0,
                'std': float(np.std(edge_counts)) if edge_counts else 0.0,
                'min': int(np.min(edge_counts)) if edge_counts else 0,
                'max': int(np.max(edge_counts)) if edge_counts else 0,
            },
        }

        logger.info(f"[方案B] 异质性分析报告:")
        logger.info(f"  H 均值={analysis['heterogeneity_distribution']['mean']:.3f}, "
                     f"标准差={analysis['heterogeneity_distribution']['std']:.3f}")
        logger.info(f"  w_spatial 均值={analysis['weight_distribution']['w_spatial_mean']:.3f}")
        logger.info(f"  w_temporal 均值={analysis['weight_distribution']['w_temporal_mean']:.3f}")
        logger.info(f"  平均边数={analysis['edge_count_stats']['mean']:.1f}")

        return graphs, analysis

    def _crop_cube(self, tc, hc, wc, radius):
        """裁剪时空子立方体（与 v2 逻辑一致）"""
        t0 = max(0, tc - radius)
        t1 = min(self.T, tc + radius + 1)
        h0 = max(0, hc - radius)
        h1 = min(self.H, hc + radius + 1)
        w0 = max(0, wc - radius)
        w1 = min(self.W, wc + radius + 1)
        return (self.data[t0:t1, h0:h1, w0:w1].copy(),
                self.valid_mask[t0:t1, h0:h1, w0:w1].copy())

    def _build_from_template(self, tc, hc, wc, template):
        """从缓存模板快速构建（与 v2 逻辑一致，但添加异质性信息）"""
        radius = template.radius
        cube, _ = self._crop_cube(tc, hc, wc, radius)
        ct = cube.shape[0] // 2
        ch = cube.shape[1] // 2
        cw = cube.shape[2] // 2

        features = np.array([
            cube[off[0] + ct, off[1] + ch, off[2] + cw]
            for off in template.node_offsets
        ], dtype=np.float32) / self.config.data.feature_scale

        # 计算异质性信息（即使从缓存构建也重新计算）
        valid_cube = self.valid_mask[
            max(0, tc - radius):min(self.T, tc + radius + 1),
            max(0, hc - radius):min(self.H, hc + radius + 1),
            max(0, wc - radius):min(self.W, wc + radius + 1)
        ].copy()

        H_val, ws, wt = self.heterogeneity_analyzer.analyze_cube(cube, valid_cube)

        return SubGraph(
            center_pos=np.array([tc, hc, wc], dtype=np.int32),
            node_features=features,
            edge_index_src=template.edge_src.copy(),
            edge_index_dst=template.edge_dst.copy(),
            edge_attrs=template.edge_attrs / self.config.data.edge_scale,
            center_value=float(self.data[tc, hc, wc]),
            num_nodes=len(template.node_offsets),
            heterogeneity_index=H_val,
            w_spatial=ws,
            w_temporal=wt,
        )
