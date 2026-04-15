"""
主控调度模块（v2）
==================
与参考实现 build_dataset.py 对齐。

支持两种模式：
  - train: 自然断点采样 → 并行构建 → 每类别保存 pkl
  - predict: 逐天 NaN 提取 → 并行构建 → 每天保存 pkl
"""

import os
import sys
import time
import pickle
import logging
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

from .config import Config
from .data_loader import NTLDataLoader
from .graph_builder import GraphBuilder


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding='utf-8')]
    )
    return logging.getLogger(__name__)


def _build_graphs_for_positions(builder, positions, desc="构建图", log=None):
    """
    为一组位置构建子图，带进度日志。
    """
    graphs = []
    total = len(positions)
    for i, (tc, hc, wc) in enumerate(positions):
        graph = builder.build_single(int(tc), int(hc), int(wc))
        if graph is not None:
            graphs.append(graph)
        if (i + 1) % 10000 == 0 and log:
            log.info(f"{desc} 进度: {i+1}/{total}, 已构建: {len(graphs)}")
    return graphs


def run_train(config):
    """
    训练模式：自然断点采样 → 并行构建 → 每类别保存 pkl。
    精确复制参考实现的 train_data_generate 逻辑。
    """
    logger = setup_logging(config.output_dir)
    logger.info("=" * 60)
    logger.info("v2 - 训练数据模式 (自然断点采样)")
    logger.info("=" * 60)

    # 加载数据
    logger.info("[1/3] 加载并预处理数据...")
    t0 = time.time()
    loader = NTLDataLoader(config)
    loader.load(
        path=config.input_path,
        quality_path=config.data.quality_path if config.data.quality_path else None,
    )

    # 自然断点采样
    logger.info("[2/3] 自然断点采样...")
    sample_results = loader.get_natural_breaks_samples()

    # 创建图构建器
    builder = GraphBuilder(config, loader.data)

    # 逐类别构建并保存
    logger.info("[3/3] 逐类别构建子图...")
    graph_cfg = config.graph
    for class_idx, positions in sample_results:
        if len(positions) == 0:
            continue

        t1 = time.time()
        class_graphs = _build_graphs_for_positions(
            builder, positions,
            desc=f"类别 {class_idx + 1}",
            log=logger
        )
        build_time = time.time() - t1

        # 保存：文件名格式与参考一致
        file_name = f"graph_{graph_cfg.sample_per_class}_{graph_cfg.search_node}_{class_idx + 1}_forTrain.pkl"
        file_path = os.path.join(config.output_dir, "train")
        os.makedirs(file_path, exist_ok=True)
        save_path = os.path.join(file_path, file_name)

        with open(save_path, 'wb') as f:
            pickle.dump(class_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            f"类别 {class_idx + 1}: 采样={len(positions)}, "
            f"有效图={len(class_graphs)}, "
            f"速度={len(class_graphs)/max(build_time, 0.001):.0f}图/秒, "
            f"保存={save_path}"
        )

    logger.info("=" * 60)
    logger.info(f"训练数据生成完成! 总耗时={time.time()-t0:.1f}s")
    logger.info("=" * 60)


def run_predict(config):
    """
    预测模式：逐天 NaN 提取 → 并行构建 → 每天保存 pkl。
    精确复制参考实现的 predict_data_generate 逻辑。
    """
    logger = setup_logging(config.output_dir)
    logger.info("=" * 60)
    logger.info("v2 - 预测数据模式 (逐天 NaN 处理)")
    logger.info("=" * 60)

    # 加载数据
    logger.info("[1/3] 加载并预处理数据...")
    t0 = time.time()
    loader = NTLDataLoader(config)
    loader.load(
        path=config.input_path,
        quality_path=config.data.quality_path if config.data.quality_path else None,
    )

    # 逐天 NaN 提取
    logger.info("[2/3] 逐天 NaN 位置提取...")
    day_results = loader.get_nan_positions_by_day()

    # 创建图构建器
    builder = GraphBuilder(config, loader.data)

    # 逐天构建并保存
    logger.info("[3/3] 逐天构建子图...")
    graph_cfg = config.graph
    for day_num, positions in day_results:
        if len(positions) == 0:
            logger.info(f"Day {day_num + 1}: 无 NaN 位置, 跳过")
            continue

        t1 = time.time()
        day_graphs = _build_graphs_for_positions(
            builder, positions,
            desc=f"Day {day_num + 1}",
            log=logger
        )
        build_time = time.time() - t1

        # 保存：文件名格式与参考一致
        file_name = f"graph_{graph_cfg.sample_per_class}_{graph_cfg.search_node}_forPred_{str(day_num + 1).zfill(3)}.pkl"
        file_path = os.path.join(config.output_dir, "predict",
                                  f"{graph_cfg.sample_per_class}_{graph_cfg.search_node}")
        os.makedirs(file_path, exist_ok=True)
        save_path = os.path.join(file_path, file_name)

        with open(save_path, 'wb') as f:
            pickle.dump(day_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            f"Day {day_num + 1}: NaN位置={len(positions)}, "
            f"有效图={len(day_graphs)}, "
            f"速度={len(day_graphs)/max(build_time, 0.001):.0f}图/秒"
        )

    logger.info("=" * 60)
    logger.info(f"预测数据生成完成! 总耗时={time.time()-t0:.1f}s")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="NTL GNN 子图构建 (v2 - 与参考实现对齐)"
    )
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'predict'],
        help='运行模式: train=自然断点采样训练数据, predict=逐天NaN预测数据'
    )
    parser.add_argument('--input', type=str, required=True, help='输入数据路径 (.npy)')
    parser.add_argument('--output', type=str, default='./output_graphs_v2', help='输出目录')
    parser.add_argument('--quality', type=str, default='', help='质量标志数据路径 (.npy)')
    parser.add_argument('--workers', type=int, default=8, help='并行进程数')
    parser.add_argument('--search-node', type=int, default=32, help='邻居节点数 (search_node)')
    parser.add_argument('--ext-range', type=int, default=6, help='子立方体提取范围 (EXT_RANGE)')
    parser.add_argument('--sample-per-class', type=int, default=20000, help='每类别采样数')
    parser.add_argument('--edge-scale', type=float, default=8.0, help='边属性归一化因子')
    parser.add_argument('--edge-time', type=int, default=50, help='有效区域时间缓冲')
    parser.add_argument('--edge-height', type=int, default=50, help='有效区域高度缓冲')
    parser.add_argument('--edge-width', type=int, default=50, help='有效区域宽度缓冲')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--no-numba', action='store_true', help='禁用 Numba JIT（纯 Python 回退）')
    args = parser.parse_args()

    config = Config()
    config.input_path = args.input
    config.output_dir = args.output
    config.data.quality_path = args.quality
    config.accel.num_workers = args.workers
    config.accel.use_numba = not args.no_numba
    config.graph.search_node = args.search_node
    config.graph.ext_range = args.ext_range
    config.graph.sample_per_class = args.sample_per_class
    config.data.edge_scale = args.edge_scale
    config.data.edge_time = args.edge_time
    config.data.edge_height = args.edge_height
    config.data.edge_width = args.edge_width
    config.seed = args.seed

    # 设置随机种子
    np.random.seed(config.seed)

    if args.mode == 'train':
        run_train(config)
    else:
        run_predict(config)


if __name__ == '__main__':
    main()
