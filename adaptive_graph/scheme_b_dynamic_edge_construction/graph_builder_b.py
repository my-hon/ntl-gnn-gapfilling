"""
方案B图构建器模块
=================
空间异质性感知的动态连边图构建器。

与 v2 GraphBuilder 的区别：
  1. 节点选择 + 边构建：复用 v2 的 GraphBuilder（round-robin + 三种边类型）
  2. 异质性分析：在图构建后计算局部异质性指数
  3. 动态权重：根据异质性指数记录 w_spatial, w_temporal
  4. SubGraph 接口：扩展了异质性字段，基础接口兼容

灵感来源：
  - Graph WaveNet (arXiv:1906.00121): 自适应邻接矩阵
  - DRTR (arXiv:2406.17281): 距离感知拓扑精炼
"""

import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from ntl_graph_accel_v2.config import Config as V2Config, GraphConfig
from ntl_graph_accel_v2.graph_builder import GraphBuilder

from .config_b import ConfigB
from .heterogeneity_analyzer import HeterogeneityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SubGraph:
    """子图数据结构（与 v2 兼容）

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
      1. 对每个中心位置，使用 v2 GraphBuilder 构建基础图
      2. 计算局部异质性指数 H = std(NTL) / mean(NTL)
      3. 根据异质性计算动态时空权重 w_spatial, w_temporal
      4. 返回带有异质性扩展字段的 SubGraph

    与 v2 GraphBuilder 的接口差异：
      - 构造函数接受 ConfigB（而非 Config）
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

        # ---- 创建 v2 GraphBuilder ----
        v2_config = self._make_v2_config(config)
        self.v2_builder = GraphBuilder(v2_config, data)

        # ---- 异质性分析器 ----
        self.heterogeneity_analyzer = HeterogeneityAnalyzer(
            cube_radius=het_cfg.heterogeneity_cube_radius,
            h_threshold=het_cfg.h_threshold,
            h_scale=het_cfg.h_scale,
            min_valid=het_cfg.min_valid_for_heterogeneity,
        )

    @staticmethod
    def _make_v2_config(config: ConfigB) -> V2Config:
        """将 ConfigB 转换为 v2 Config"""
        return V2Config(
            data=config.data,
            graph=GraphConfig(
                search_node=config.graph.search_node,
                ext_range=config.graph.ext_range,
                max_ext=config.graph.max_ext,
                num_regions=config.graph.num_regions,
            ),
            accel=config.accel,
            output_dir=config.output_dir,
            cache_dir=config.cache_dir,
            seed=config.seed,
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
        # 使用 v2 GraphBuilder 构建基础图
        result = self.v2_builder.build_single(tc, hc, wc)
        if result is None:
            return None

        # ---- 方案B核心：计算异质性指数 ----
        H_val, ws, wt = self._compute_heterogeneity(tc, hc, wc)

        # 从 v2 dict 提取边信息
        edge_index = result['edge_index']  # (2, E)
        edge_src = edge_index[0].astype(np.int64)
        edge_dst = edge_index[1].astype(np.int64)
        edge_attrs = result['edge_attr']
        node_features = result['node_features']

        center_value = float(self.data[tc, hc, wc])

        # 构建方案B的 SubGraph（包含异质性信息）
        subgraph = SubGraph(
            center_pos=np.array([tc, hc, wc], dtype=np.int32),
            node_features=node_features,
            edge_index_src=edge_src,
            edge_index_dst=edge_dst,
            edge_attrs=edge_attrs,
            center_value=center_value,
            num_nodes=node_features.shape[0],
            heterogeneity_index=H_val,
            w_spatial=ws,
            w_temporal=wt,
        )

        return subgraph

    def _compute_heterogeneity(self, tc: int, hc: int, wc: int) -> Tuple[float, float, float]:
        """
        计算指定位置的异质性指数和动态权重。

        Parameters
        ----------
        tc, hc, wc : int
            中心位置坐标

        Returns
        -------
        H_val : float
            异质性指数
        w_spatial : float
            空间权重
        w_temporal : float
            时序权重
        """
        het_cfg = self.config.heterogeneity
        radius = het_cfg.heterogeneity_cube_radius

        # 裁剪局部立方体
        t0 = max(0, tc - radius)
        t1 = min(self.T, tc + radius + 1)
        h0 = max(0, hc - radius)
        h1 = min(self.H, hc + radius + 1)
        w0 = max(0, wc - radius)
        w1 = min(self.W, wc + radius + 1)

        cube = self.data[t0:t1, h0:h1, w0:w1].copy()
        valid_cube = self.valid_mask[t0:t1, h0:h1, w0:w1].copy()

        H_val, ws, wt = self.heterogeneity_analyzer.analyze_cube(cube, valid_cube)
        return H_val, ws, wt

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
                logger.info(
                    f"[方案B] 进度: {i+1}/{total}, 已构建: {len(graphs)}, "
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
