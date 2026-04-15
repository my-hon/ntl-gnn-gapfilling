"""
主控调度模块（v2）
==================
"""

import os
import sys
import time
import pickle
import logging
import argparse
import numpy as np
from datetime import datetime

from .config import Config
from .data_loader import NTLDataLoader
from .spatial_partitioner import ParallelGraphProcessor


def setup_logging(output_dir):
    log_file = os.path.join(output_dir, f"build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding='utf-8')]
    )
    return logging.getLogger(__name__)


def build_missing_graphs(config):
    logger = setup_logging(config.output_dir)
    logger.info("=" * 60)
    logger.info("v2 (Numba JIT) - 缺失值填补模式")
    logger.info("=" * 60)

    logger.info("[1/4] 加载数据...")
    t0 = time.time()
    loader = NTLDataLoader(config)
    loader.load(path=config.input_path)

    logger.info("[2/4] 识别缺失位置...")
    positions = loader.get_missing_positions()
    if len(positions) == 0:
        logger.warning("未发现缺失位置")
        return

    logger.info(f"[3/4] 并行构建子图 ({config.accel.num_workers}进程)...")
    t1 = time.time()
    processor = ParallelGraphProcessor(config)
    graphs = processor.process(loader.data, loader.valid_mask, positions, mode="missing")
    build_time = time.time() - t1

    logger.info("[4/4] 保存结果...")
    output_path = os.path.join(config.output_dir, "missing_graphs.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("=" * 60)
    logger.info(f"完成! 缺失={len(positions)}, 构建={len(graphs)}, "
                f"耗时={time.time()-t0:.1f}s, 速度={len(graphs)/max(build_time,0.001):.0f}图/秒")
    logger.info("=" * 60)


def build_training_graphs(config):
    logger = setup_logging(config.output_dir)
    logger.info("=" * 60)
    logger.info("v2 (Numba JIT) - 训练数据模式")
    logger.info("=" * 60)

    logger.info("[1/5] 加载数据...")
    t0 = time.time()
    loader = NTLDataLoader(config)
    loader.load(path=config.input_path)

    logger.info("[2/5] 识别有效位置...")
    all_pos = loader.get_all_valid_positions()

    num_samples = min(150000, len(all_pos))
    logger.info(f"[3/5] 采样 {num_samples}...")
    rng = np.random.RandomState(config.seed)
    positions = all_pos[rng.choice(len(all_pos), size=num_samples, replace=False)]

    logger.info(f"[4/5] 并行构建 ({config.accel.num_workers}进程)...")
    t1 = time.time()
    processor = ParallelGraphProcessor(config)
    graphs = processor.process(loader.data, loader.valid_mask, positions, mode="all")
    build_time = time.time() - t1

    logger.info("[5/5] 划分数据集...")
    rng.shuffle(graphs)
    n = len(graphs)
    splits = {
        'train': graphs[:int(n*0.6)],
        'val': graphs[int(n*0.6):int(n*0.8)],
        'test': graphs[int(n*0.8):]
    }
    for name, gs in splits.items():
        path = os.path.join(config.output_dir, f"{name}_graphs.pkl")
        with open(path, 'wb') as f:
            pickle.dump(gs, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"  {name}: {len(gs)} -> {path}")

    logger.info("=" * 60)
    logger.info(f"完成! 采样={num_samples}, 构建={len(graphs)}, "
                f"速度={len(graphs)/max(build_time,0.001):.0f}图/秒")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="NTL GNN 子图加速构建 (v2 Numba JIT)")
    parser.add_argument('--mode', type=str, default='missing', choices=['missing', 'training'])
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='./output_graphs_v2')
    parser.add_argument('--cache-dir', type=str, default='./graph_cache_v2')
    parser.add_argument('--tile-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--use-cache', action='store_true', default=True)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--cache-quant', type=int, default=2)
    parser.add_argument('--num-nodes', type=int, default=36)
    parser.add_argument('--initial-radius', type=int, default=4)
    parser.add_argument('--max-radius', type=int, default=20)
    parser.add_argument('--buffer-size', type=int, default=50)
    parser.add_argument('--temporal-buffer', type=int, default=10)
    parser.add_argument('--no-numba', action='store_true', help='禁用 Numba JIT（纯 Python 回退）')
    args = parser.parse_args()

    config = Config()
    config.input_path = args.input
    config.output_dir = args.output
    config.cache_dir = args.cache_dir
    config.accel.tile_size = args.tile_size
    config.accel.num_workers = args.workers
    config.accel.use_cache = args.use_cache and not args.no_cache
    config.accel.cache_quantization = args.cache_quant
    config.accel.use_numba = not args.no_numba
    config.graph.num_nodes = args.num_nodes
    config.graph.initial_radius = args.initial_radius
    config.graph.max_radius = args.max_radius
    config.data.buffer_size = args.buffer_size
    config.data.temporal_buffer = args.temporal_buffer

    if args.mode == 'missing':
        build_missing_graphs(config)
    else:
        build_training_graphs(config)


if __name__ == '__main__':
    main()
