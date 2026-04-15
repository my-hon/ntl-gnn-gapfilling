"""
方案A图构建器 - 数据质量驱动的自适应节点数
=============================================
包装 v2 的 GraphBuilder，在 build_single 前根据数据质量动态调整 search_node。

核心改动：
  - build_single: 先查询质量分数，动态决定节点数，临时修改 v2 builder 的 search_node
  - 保留 v2 的 round-robin 象限选择、三种边类型等核心算法
  - SubGraph 数据结构保持兼容

与 v2 的接口兼容性：
  - SubGraph 数据结构完全不变
  - 输出格式完全不变
"""

import numpy as np
import logging
from typing import Optional, Tuple, List

from .config_a import ConfigA
from .adaptive_selector import AdaptiveNodeSelector

# 复用 v2 的基础设施
from ntl_graph_accel_v2.config import Config as V2Config, GraphConfig
from ntl_graph_accel_v2.graph_builder import GraphBuilder, SubGraph

logger = logging.getLogger(__name__)


class GraphBuilderA:
    """
    方案A：数据质量驱动的自适应图构建器。

    与 v2 GraphBuilder 的区别：
    - 节点数不再固定，而是根据局部数据质量动态调整
    - 高质量区域（有效像素密集、空间连续、时序稳定）→ 更多节点
    - 低质量区域（稀疏、断裂、波动大）→ 更少节点
    - 减少低质量区域的冗余计算，提高高质量区域的表示精度

    实现方式：包装 v2 GraphBuilder，在每次 build_single 前临时修改
    config.graph.search_node 为自适应节点数，构建完成后恢复原值。
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

        # ---- 创建 v2 GraphBuilder ----
        v2_config = self._make_v2_config(config)
        self.v2_builder = GraphBuilder(v2_config, data)
        self.original_search_node = v2_config.graph.search_node

        # ---- 统计信息 ----
        self._stats = {
            'total_queries': 0,
            'adaptive_counts': {},  # 节点数 → 出现次数
        }

    @staticmethod
    def _make_v2_config(config: ConfigA) -> V2Config:
        """将 ConfigA 转换为 v2 Config"""
        return V2Config(
            data=config.data,
            graph=GraphConfig(
                search_node=config.graph.num_nodes_base,
                ext_range=config.graph.initial_radius,
                max_ext=config.graph.max_radius,
                num_regions=config.graph.num_regions,
            ),
            accel=config.accel,
            output_dir=config.output_dir,
            cache_dir=config.cache_dir,
            seed=config.seed,
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
        2. 临时修改 v2 builder 的 search_node 为自适应值
        3. 调用 v2 builder 的 build_single 得到 dict，转换为 SubGraph
        4. 恢复原始 search_node

        Parameters
        ----------
        tc, hc, wc : int
            时空位置坐标

        Returns
        -------
        SubGraph 或 None
        """
        # ---- 方案A核心：查询自适应节点数 ----
        num_nodes, quality = self.get_num_nodes(hc, wc)

        # 临时修改 v2 builder 的 search_node
        self.v2_builder.config.graph.search_node = num_nodes

        try:
            # 调用 v2 builder 构建（返回 dict）
            result = self.v2_builder.build_single(tc, hc, wc)
        finally:
            # 恢复原始 search_node
            self.v2_builder.config.graph.search_node = self.original_search_node

        if result is None:
            return None

        # 将 v2 dict 转换为 SubGraph
        subgraph = SubGraph(
            center_pos=np.array([tc, hc, wc], dtype=np.int32),
            node_features=result['node_features'],
            edge_index=result['edge_index'],
            edge_attr=result['edge_attr'],
            ground_truth=result['ground_truth'],
            position=result['position'],
            num_nodes=result['node_features'].shape[0],
        )

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
                logger.info(f"[方案A] 进度: {i+1}/{total}, "
                             f"已构建: {len(graphs)}")

        # 打印统计信息
        self._print_stats(total, len(graphs))

        return graphs

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
        }
