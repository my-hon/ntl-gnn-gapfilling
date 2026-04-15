"""
空间分块并行化模块
==================
将研究区域划分为独立瓦片，使用多进程并行处理。
每个瓦片内的子图构建相互独立，可安全并行。
"""

import os
import pickle
import logging
import multiprocessing as mp
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from functools import partial

logger = logging.getLogger(__name__)


@dataclass
class Tile:
    """空间瓦片定义"""
    tile_id: int
    # 空间范围（含重叠区域）
    h_start: int
    h_end: int
    w_start: int
    w_end: int
    # 有效范围（不含重叠）
    h_valid_start: int
    h_valid_end: int
    w_valid_start: int
    w_valid_end: int


class SpatialPartitioner:
    """
    空间分块并行处理器。
    
    工作流程：
    1. 将空间区域划分为不重叠瓦片
    2. 每个瓦片添加重叠区域（用于边界节点构建）
    3. 多进程并行处理各瓦片
    4. 汇总结果，裁剪重叠区域产生的冗余子图
    """

    def __init__(
        self,
        data: np.ndarray,
        valid_mask: np.ndarray,
        tile_size: int = 128,
        num_workers: int = 8,
        overlap: int = 20
    ):
        """
        Parameters
        ----------
        data : np.ndarray
            完整NTL数据 (T, H, W)
        valid_mask : np.ndarray
            有效像素掩码 (T, H, W)
        tile_size : int
            瓦片大小（像素）
        num_workers : int
            并行进程数
        overlap : int
            瓦片间重叠区域大小（应 >= 图构建最大半径）
        """
        self.data = data
        self.valid_mask = valid_mask
        self.T, self.H, self.W = data.shape
        self.tile_size = tile_size
        self.num_workers = num_workers
        self.overlap = overlap

    def partition(self) -> List[Tile]:
        """
        将空间区域划分为瓦片。

        Returns
        -------
        List[Tile]
            瓦片列表
        """
        tiles = []
        tile_id = 0

        h = 0
        while h < self.H:
            w = 0
            while w < self.W:
                # 瓦片范围（含重叠）
                h_start = max(0, h - self.overlap)
                h_end = min(self.H, h + self.tile_size + self.overlap)
                w_start = max(0, w - self.overlap)
                w_end = min(self.W, w + self.tile_size + self.overlap)

                # 有效范围（不含重叠）
                h_valid_start = h
                h_valid_end = min(self.H, h + self.tile_size)
                w_valid_start = w
                w_valid_end = min(self.W, w + self.tile_size)

                tiles.append(Tile(
                    tile_id=tile_id,
                    h_start=h_start, h_end=h_end,
                    w_start=w_start, w_end=w_end,
                    h_valid_start=h_valid_start, h_valid_end=h_valid_end,
                    w_valid_start=w_valid_start, w_valid_end=w_valid_end
                ))

                tile_id += 1
                w += self.tile_size

            h += self.tile_size

        logger.info(f"空间分块完成: {len(tiles)} 个瓦片, "
                     f"瓦片大小={self.tile_size}, 重叠={self.overlap}")
        return tiles

    def get_tile_data(self, tile: Tile) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取瓦片对应的数据切片。

        Returns
        -------
        tile_data : np.ndarray
            瓦片NTL数据 (T, tile_h, tile_w)
        tile_mask : np.ndarray
            瓦片有效掩码
        """
        tile_data = self.data[:, tile.h_start:tile.h_end, tile.w_start:tile.w_end].copy()
        tile_mask = self.valid_mask[:, tile.h_start:tile.h_end, tile.w_start:tile.w_end].copy()
        return tile_data, tile_mask

    def get_tile_positions(
        self, tile: Tile, positions: np.ndarray
    ) -> np.ndarray:
        """
        获取属于指定瓦片有效范围内的位置。

        Parameters
        ----------
        tile : Tile
        positions : np.ndarray
            全局坐标数组 (N, 3)

        Returns
        -------
        np.ndarray
            属于该瓦片的位置
        """
        mask = (
            (positions[:, 1] >= tile.h_valid_start) &
            (positions[:, 1] < tile.h_valid_end) &
            (positions[:, 2] >= tile.w_valid_start) &
            (positions[:, 2] < tile.w_valid_end)
        )
        return positions[mask]


def _worker_process_tile(
    tile: Tile,
    positions: np.ndarray,
    full_data: np.ndarray,
    full_mask: np.ndarray,
    config_dict: dict,
    output_dir: str
):
    """
    单个瓦片的工作函数（在子进程中执行）。
    
    Parameters
    ----------
    tile : Tile
        瓦片定义
    positions : np.ndarray
        该瓦片内的位置坐标
    full_data : np.ndarray
        完整数据（用于裁剪立方体）
    full_mask : np.ndarray
        完整掩码
    config_dict : dict
        配置字典（可pickle序列化）
    output_dir : str
        输出目录
    """
    # 延迟导入避免fork问题
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from ntl_graph_accel.config import Config, DataConfig, GraphConfig, AccelerationConfig
    from ntl_graph_accel.graph_builder import GraphBuilder

    # 重建配置
    config = Config(
        data=DataConfig(**config_dict['data']),
        graph=GraphConfig(**config_dict['graph']),
        accel=AccelerationConfig(**config_dict['accel']),
        output_dir=output_dir,
        cache_dir=config_dict.get('cache_dir', './graph_cache')
    )

    # 获取瓦片数据
    tile_data = full_data[:, tile.h_start:tile.h_end, tile.w_start:tile.w_end].copy()
    tile_mask = full_mask[:, tile.h_start:tile.h_end, tile.w_start:tile.w_end].copy()

    # 将全局坐标转换为瓦片局部坐标
    local_positions = positions.copy()
    local_positions[:, 1] -= tile.h_start
    local_positions[:, 2] -= tile.w_start

    # 构建图构建器（新 GraphBuilder 只接受 config 和 data 两个参数）
    builder = GraphBuilder(config, tile_data)

    # 批量构建子图
    graphs = []
    for i, (tc, hc, wc) in enumerate(local_positions):
        tc, hc, wc = int(tc), int(hc), int(wc)
        graph = builder.build_single(tc, hc, wc)
        if graph is not None:
            # 将局部坐标转回全局坐标（graph 是 dict 格式）
            if 'position' in graph:
                graph['position'][0, 1] += tile.h_start
                graph['position'][0, 2] += tile.w_start
            graphs.append(graph)

    # 保存结果
    output_path = os.path.join(output_dir, f"tile_{tile.tile_id:04d}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 返回统计信息
    cache_stats = builder.cache.get_stats() if builder.cache else {}
    return {
        'tile_id': tile.tile_id,
        'num_positions': len(positions),
        'num_graphs': len(graphs),
        'cache_hit_rate': cache_stats.get('hit_rate', 0.0),
        'output_path': output_path
    }


class ParallelGraphProcessor:
    """
    并行图构建处理器。
    整合空间分块、多进程并行、缓存复用。
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : Config
            全局配置
        """
        self.config = config

    def process(
        self,
        data: np.ndarray,
        valid_mask: np.ndarray,
        positions: np.ndarray,
        mode: str = "missing"
    ) -> List:
        """
        并行处理所有位置的子图构建。

        Parameters
        ----------
        data : np.ndarray
            完整NTL数据 (T, H, W)
        valid_mask : np.ndarray
            有效像素掩码
        positions : np.ndarray
            需要处理的位置 (N, 3)
        mode : str
            "missing" - 仅处理缺失位置
            "all" - 处理所有有效位置（训练数据）

        Returns
        -------
        List[SubGraph]
            所有构建的子图
        """
        accel = self.config.accel
        output_dir = self.config.output_dir

        # 创建分块器
        partitioner = SpatialPartitioner(
            data=data,
            valid_mask=valid_mask,
            tile_size=accel.tile_size,
            num_workers=accel.num_workers,
            overlap=self.config.graph.max_radius + 2
        )

        # 划分瓦片
        tiles = partitioner.partition()

        # 为每个瓦片分配位置
        tile_positions = []
        for tile in tiles:
            tile_pos = partitioner.get_tile_positions(tile, positions)
            if len(tile_pos) > 0:
                tile_positions.append((tile, tile_pos))

        logger.info(f"共 {len(positions)} 个位置, "
                     f"分配到 {len(tile_positions)} 个非空瓦片")

        # 序列化配置（multiprocessing需要）
        config_dict = {
            'data': {
                'data_shape': self.config.data.data_shape,
                'buffer_size': self.config.data.buffer_size,
                'temporal_buffer': self.config.data.temporal_buffer,
                'feature_scale': self.config.data.feature_scale,
                'edge_scale': self.config.data.edge_scale,
                'ext_range': self.config.data.ext_range,
                'search_node': self.config.data.search_node,
                'natural_breaks': self.config.data.natural_breaks,
                'sample_per_class': self.config.data.sample_per_class,
                'edge_time': self.config.data.edge_time,
                'edge_height': self.config.data.edge_height,
                'edge_width': self.config.data.edge_width,
                'quality_path': self.config.data.quality_path
            },
            'graph': {
                'num_nodes': self.config.graph.num_nodes,
                'initial_radius': self.config.graph.initial_radius,
                'max_radius': self.config.graph.max_radius,
                'num_regions': self.config.graph.num_regions,
                'max_bresenham_len': self.config.graph.max_bresenham_len
            },
            'accel': {
                'tile_size': self.config.accel.tile_size,
                'num_workers': self.config.accel.num_workers,
                'use_cuda': False,  # 子进程不使用GPU（避免CUDA上下文问题）
                'gpu_batch_size': self.config.accel.gpu_batch_size,
                'bresenham_lookup': self.config.accel.bresenham_lookup,
                'use_cache': self.config.accel.use_cache,
                'cache_quantization': self.config.accel.cache_quantization,
                'cache_max_size': self.config.accel.cache_max_size,
                'output_format': self.config.accel.output_format,
                'save_per_tile': self.config.accel.save_per_tile
            },
            'cache_dir': self.config.cache_dir
        }

        # 多进程并行处理
        all_graphs = []
        stats_list = []

        if accel.num_workers > 1 and len(tile_positions) > 1:
            # 多进程模式
            ctx = mp.get_context('spawn')  # 使用spawn避免fork问题
            pool = ctx.Pool(processes=min(accel.num_workers, len(tile_positions)))

            results = []
            for tile, tile_pos in tile_positions:
                result = pool.apply_async(
                    _worker_process_tile,
                    args=(tile, tile_pos, data, valid_mask, config_dict, output_dir)
                )
                results.append(result)

            pool.close()
            pool.join()

            for result in results:
                try:
                    stat = result.get(timeout=3600)
                    stats_list.append(stat)
                except Exception as e:
                    logger.error(f"瓦片处理失败: {e}")

            # 加载所有瓦片结果
            for stat in stats_list:
                if os.path.exists(stat['output_path']):
                    with open(stat['output_path'], 'rb') as f:
                        graphs = pickle.load(f)
                    all_graphs.extend(graphs)

        else:
            # 单进程模式
            for tile, tile_pos in tile_positions:
                stat = _worker_process_tile(
                    tile, tile_pos, data, valid_mask, config_dict, output_dir
                )
                stats_list.append(stat)
                if os.path.exists(stat['output_path']):
                    with open(stat['output_path'], 'rb') as f:
                        graphs = pickle.load(f)
                    all_graphs.extend(graphs)

        # 打印统计
        total_positions = sum(s['num_positions'] for s in stats_list)
        total_graphs = sum(s['num_graphs'] for s in stats_list)
        avg_hit_rate = np.mean([s['cache_hit_rate'] for s in stats_list])

        logger.info(f"处理完成: "
                     f"总位置={total_positions}, "
                     f"成功构建={total_graphs}, "
                     f"成功率={total_graphs/max(total_positions,1):.1%}, "
                     f"平均缓存命中率={avg_hit_rate:.1%}")

        return all_graphs

    def merge_tile_results(self, output_dir: str) -> List:
        """
        合并所有瓦片的结果文件。

        Parameters
        ----------
        output_dir : str
            瓦片结果所在目录

        Returns
        -------
        List[SubGraph]
            合并后的所有子图
        """
        all_graphs = []
        tile_files = sorted([
            f for f in os.listdir(output_dir) if f.startswith("tile_") and f.endswith(".pkl")
        ])

        for f in tile_files:
            path = os.path.join(output_dir, f)
            with open(path, 'rb') as fp:
                graphs = pickle.load(fp)
            all_graphs.extend(graphs)

        logger.info(f"合并完成: {len(tile_files)} 个瓦片, 共 {len(all_graphs)} 个子图")
        return all_graphs
