"""
空间分块并行化模块（v2）
==========================
与 v1 逻辑一致，适配 v2 的 GraphBuilder 接口。
"""

import os
import pickle
import logging
import multiprocessing as mp
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Tile:
    tile_id: int
    h_start: int
    h_end: int
    w_start: int
    w_end: int
    h_valid_start: int
    h_valid_end: int
    w_valid_start: int
    w_valid_end: int


class SpatialPartitioner:
    """空间分块并行处理器"""

    def __init__(self, data, valid_mask, tile_size=128, num_workers=8, overlap=20):
        self.data = data
        self.valid_mask = valid_mask
        self.T, self.H, self.W = data.shape
        self.tile_size = tile_size
        self.num_workers = num_workers
        self.overlap = overlap

    def partition(self) -> List[Tile]:
        tiles = []
        tile_id = 0
        h = 0
        while h < self.H:
            w = 0
            while w < self.W:
                h_start = max(0, h - self.overlap)
                h_end = min(self.H, h + self.tile_size + self.overlap)
                w_start = max(0, w - self.overlap)
                w_end = min(self.W, w + self.tile_size + self.overlap)
                tiles.append(Tile(
                    tile_id=tile_id,
                    h_start=h_start, h_end=h_end,
                    w_start=w_start, w_end=w_end,
                    h_valid_start=h, h_valid_end=min(self.H, h + self.tile_size),
                    w_valid_start=w, w_valid_end=min(self.W, w + self.tile_size),
                ))
                tile_id += 1
                w += self.tile_size
            h += self.tile_size
        logger.info(f"空间分块: {len(tiles)} 个瓦片")
        return tiles

    def get_tile_positions(self, tile, positions):
        mask = (
            (positions[:, 1] >= tile.h_valid_start) &
            (positions[:, 1] < tile.h_valid_end) &
            (positions[:, 2] >= tile.w_valid_start) &
            (positions[:, 2] < tile.w_valid_end)
        )
        return positions[mask]


def _worker_process_tile(tile, positions, full_data, full_mask, config_dict, output_dir):
    """子进程工作函数"""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ntl_graph_accel_v2.config import Config, DataConfig, GraphConfig, AccelerationConfig
    from ntl_graph_accel_v2.graph_builder import GraphBuilder

    config = Config(
        data=DataConfig(**config_dict['data']),
        graph=GraphConfig(**config_dict['graph']),
        accel=AccelerationConfig(**config_dict['accel']),
        output_dir=output_dir,
        cache_dir=config_dict.get('cache_dir', './graph_cache_v2')
    )

    tile_data = full_data[:, tile.h_start:tile.h_end, tile.w_start:tile.w_end].copy()
    tile_mask = full_mask[:, tile.h_start:tile.h_end, tile.w_start:tile.w_end].copy()

    local_pos = positions.copy()
    local_pos[:, 1] -= tile.h_start
    local_pos[:, 2] -= tile.w_start

    builder = GraphBuilder(config, tile_data, tile_mask)

    graphs = []
    for tc, hc, wc in local_pos:
        g = builder.build_single(int(tc), int(hc), int(wc))
        if g is not None:
            g.center_pos[1] += tile.h_start
            g.center_pos[2] += tile.w_start
            graphs.append(g)

    path = os.path.join(output_dir, f"tile_{tile.tile_id:04d}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    stats = builder.cache.get_stats() if builder.cache else {}
    return {'tile_id': tile.tile_id, 'num_positions': len(positions),
            'num_graphs': len(graphs), 'cache_hit_rate': stats.get('hit_rate', 0.0),
            'output_path': path}


class ParallelGraphProcessor:
    """并行图构建处理器"""

    def __init__(self, config):
        self.config = config

    def process(self, data, valid_mask, positions, mode="missing"):
        accel = self.config.accel
        output_dir = self.config.output_dir

        partitioner = SpatialPartitioner(
            data, valid_mask,
            tile_size=accel.tile_size,
            num_workers=accel.num_workers,
            overlap=self.config.graph.max_radius + 2
        )
        tiles = partitioner.partition()

        tile_positions = [(t, partitioner.get_tile_positions(t, positions)) for t in tiles]
        tile_positions = [(t, p) for t, p in tile_positions if len(p) > 0]

        logger.info(f"共 {len(positions)} 位置, 分配到 {len(tile_positions)} 瓦片")

        config_dict = {
            'data': {
                'data_shape': self.config.data.data_shape,
                'buffer_size': self.config.data.buffer_size,
                'temporal_buffer': self.config.data.temporal_buffer,
                'feature_scale': self.config.data.feature_scale,
                'edge_scale': self.config.data.edge_scale
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
                'use_numba': self.config.accel.use_numba,
                'bresenham_lookup': self.config.accel.bresenham_lookup,
                'use_cache': self.config.accel.use_cache,
                'cache_quantization': self.config.accel.cache_quantization,
                'cache_max_size': self.config.accel.cache_max_size,
                'output_format': self.config.accel.output_format,
                'save_per_tile': self.config.accel.save_per_tile
            },
            'cache_dir': self.config.cache_dir
        }

        all_graphs = []
        stats_list = []

        if accel.num_workers > 1 and len(tile_positions) > 1:
            ctx = mp.get_context('spawn')
            pool = ctx.Pool(processes=min(accel.num_workers, len(tile_positions)))
            results = [pool.apply_async(_worker_process_tile,
                        args=(t, p, data, valid_mask, config_dict, output_dir))
                       for t, p in tile_positions]
            pool.close()
            pool.join()
            for r in results:
                try:
                    stats_list.append(r.get(timeout=3600))
                except Exception as e:
                    logger.error(f"瓦片处理失败: {e}")
        else:
            for t, p in tile_positions:
                stats_list.append(_worker_process_tile(t, p, data, valid_mask, config_dict, output_dir))

        for s in stats_list:
            if os.path.exists(s['output_path']):
                with open(s['output_path'], 'rb') as f:
                    all_graphs.extend(pickle.load(f))

        total_pos = sum(s['num_positions'] for s in stats_list)
        total_g = sum(s['num_graphs'] for s in stats_list)
        avg_hr = np.mean([s['cache_hit_rate'] for s in stats_list])
        logger.info(f"完成: 位置={total_pos}, 子图={total_g}, 成功率={total_g/max(total_pos,1):.1%}, 缓存命中={avg_hr:.1%}")

        return all_graphs
