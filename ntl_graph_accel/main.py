"""
主控调度模块
============
整合所有模块，提供统一的命令行入口。
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


def setup_logging(output_dir: str):
    """配置日志"""
    log_file = os.path.join(output_dir, f"graph_build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


def build_missing_graphs(config: Config):
    """
    构建缺失位置的子图（用于推理/填补）。
    """
    logger = setup_logging(config.output_dir)
    logger.info("=" * 60)
    logger.info("模式: 缺失值填补 - 构建缺失位置子图")
    logger.info("=" * 60)

    # 1. 加载数据
    logger.info("[1/4] 加载数据...")
    t0 = time.time()
    loader = NTLDataLoader(config)
    loader.load(path=config.input_path)
    logger.info(f"数据加载耗时: {time.time()-t0:.1f}s")

    # 2. 获取缺失位置
    logger.info("[2/4] 识别缺失位置...")
    positions = loader.get_missing_positions()
    if len(positions) == 0:
        logger.warning("未发现缺失位置，无需处理")
        return

    # 3. 并行构建子图
    logger.info(f"[3/4] 并行构建子图 ({config.accel.num_workers}进程)...")
    t1 = time.time()
    processor = ParallelGraphProcessor(config)
    graphs = processor.process(
        data=loader.data,
        valid_mask=loader.valid_mask,
        positions=positions,
        mode="missing"
    )
    build_time = time.time() - t1
    logger.info(f"子图构建耗时: {build_time:.1f}s")

    # 4. 保存结果
    logger.info("[4/4] 保存结果...")
    output_path = os.path.join(config.output_dir, "missing_graphs.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 统计报告
    logger.info("=" * 60)
    logger.info("处理完成!")
    logger.info(f"  缺失位置总数: {len(positions)}")
    logger.info(f"  成功构建子图: {len(graphs)}")
    logger.info(f"  成功率: {len(graphs)/max(len(positions),1):.1%}")
    logger.info(f"  总耗时: {time.time()-t0:.1f}s")
    logger.info(f"  构建速度: {len(graphs)/max(build_time,0.001):.0f} 图/秒")
    logger.info(f"  输出文件: {output_path}")
    logger.info("=" * 60)


def build_training_graphs(config: Config):
    """
    构建训练数据子图（用于模型训练）。
    """
    logger = setup_logging(config.output_dir)
    logger.info("=" * 60)
    logger.info("模式: 训练数据 - 构建有效位置子图")
    logger.info("=" * 60)

    # 1. 加载数据
    logger.info("[1/5] 加载数据...")
    t0 = time.time()
    loader = NTLDataLoader(config)
    loader.load(path=config.input_path)
    logger.info(f"数据加载耗时: {time.time()-t0:.1f}s")

    # 2. 获取所有有效位置
    logger.info("[2/5] 识别有效位置...")
    all_positions = loader.get_all_valid_positions()

    # 3. 随机采样（论文使用150,000个训练样本）
    num_samples = min(150000, len(all_positions))
    logger.info(f"[3/5] 随机采样 {num_samples} 个位置...")
    rng = np.random.RandomState(config.seed)
    indices = rng.choice(len(all_positions), size=num_samples, replace=False)
    positions = all_positions[indices]

    # 4. 并行构建子图
    logger.info(f"[4/5] 并行构建子图 ({config.accel.num_workers}进程)...")
    t1 = time.time()
    processor = ParallelGraphProcessor(config)
    graphs = processor.process(
        data=loader.data,
        valid_mask=loader.valid_mask,
        positions=positions,
        mode="all"
    )
    build_time = time.time() - t1
    logger.info(f"子图构建耗时: {build_time:.1f}s")

    # 5. 划分训练/验证/测试集并保存
    logger.info("[5/5] 划分数据集并保存...")
    rng.shuffle(graphs)
    n = len(graphs)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    splits = {
        'train': graphs[:train_end],
        'val': graphs[train_end:val_end],
        'test': graphs[val_end:]
    }

    for split_name, split_graphs in splits.items():
        path = os.path.join(config.output_dir, f"{split_name}_graphs.pkl")
        with open(path, 'wb') as f:
            pickle.dump(split_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"  {split_name}: {len(split_graphs)} 个子图 -> {path}")

    # 统计报告
    logger.info("=" * 60)
    logger.info("处理完成!")
    logger.info(f"  有效位置总数: {len(all_positions)}")
    logger.info(f"  采样数量: {num_samples}")
    logger.info(f"  成功构建: {len(graphs)}")
    logger.info(f"  训练/验证/测试: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    logger.info(f"  总耗时: {time.time()-t0:.1f}s")
    logger.info(f"  构建速度: {len(graphs)/max(build_time,0.001):.0f} 图/秒")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="NTL数据GNN子图加速构建工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 构建缺失位置子图（用于填补）
  python -m ntl_graph_accel.main --mode missing --input data.npy --output ./output

  # 构建训练数据子图
  python -m ntl_graph_accel.main --mode training --input data.npy --output ./output

  # 自定义参数
  python -m ntl_graph_accel.main --mode missing --input data.npy \\
      --tile-size 128 --workers 8 --use-cache --cache-quant 2
        """
    )

    parser.add_argument('--mode', type=str, default='missing',
                        choices=['missing', 'training'],
                        help='运行模式: missing(缺失填补) 或 training(训练数据)')
    parser.add_argument('--input', type=str, required=True,
                        help='输入数据路径 (.npy格式)')
    parser.add_argument('--output', type=str, default='./output_graphs',
                        help='输出目录')
    parser.add_argument('--cache-dir', type=str, default='./graph_cache',
                        help='缓存目录')

    # 加速参数
    parser.add_argument('--tile-size', type=int, default=128,
                        help='空间瓦片大小（像素）')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行进程数')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='启用CUDA加速')
    parser.add_argument('--use-cache', action='store_true', default=True,
                        help='启用缓存复用')
    parser.add_argument('--no-cache', action='store_true', default=False,
                        help='禁用缓存')
    parser.add_argument('--cache-quant', type=int, default=2,
                        help='缓存量化步长')

    # 图构建参数
    parser.add_argument('--num-nodes', type=int, default=36,
                        help='图中节点数')
    parser.add_argument('--initial-radius', type=int, default=4,
                        help='初始半窗口大小')
    parser.add_argument('--max-radius', type=int, default=20,
                        help='最大半窗口大小')

    # 数据参数
    parser.add_argument('--buffer-size', type=int, default=50,
                        help='空间缓冲区大小')
    parser.add_argument('--temporal-buffer', type=int, default=10,
                        help='时间缓冲区大小（天数）')

    args = parser.parse_args()

    # 构建配置
    config = Config()
    config.input_path = args.input
    config.output_dir = args.output
    config.cache_dir = args.cache_dir

    # 加速配置
    config.accel.tile_size = args.tile_size
    config.accel.num_workers = args.workers
    config.accel.use_cuda = args.use_cuda
    config.accel.use_cache = args.use_cache and not args.no_cache
    config.accel.cache_quantization = args.cache_quant

    # 图构建配置
    config.graph.num_nodes = args.num_nodes
    config.graph.initial_radius = args.initial_radius
    config.graph.max_radius = args.max_radius

    # 数据配置
    config.data.buffer_size = args.buffer_size
    config.data.temporal_buffer = args.temporal_buffer

    # 执行
    if args.mode == 'missing':
        build_missing_graphs(config)
    else:
        build_training_graphs(config)


if __name__ == '__main__':
    main()
