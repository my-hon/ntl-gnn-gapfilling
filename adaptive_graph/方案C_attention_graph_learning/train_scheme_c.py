#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方案C 完整训练流程

用法:
  python train_scheme_c.py                    # 完整流程（生成图+训练）
  python train_scheme_c.py --skip-build       # 跳过图生成，直接训练
  python train_scheme_c.py --build-only       # 只生成图，不训练
"""

import sys
import os
import time
import logging
import pickle
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# ================================================================
# 日志配置
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger('scheme_c_train')


def load_and_preprocess(ntl_path, quality_path):
    """加载并预处理数据"""
    logger.info('[1/5] 加载并预处理数据...')
    t0 = time.time()
    raw = np.load(ntl_path)
    quality = np.load(quality_path)

    data = raw.copy().astype(np.float32)
    data[quality > 1] = np.nan
    data = data / 10.0 / 100.0

    valid_mask = ~np.isnan(data)
    T, H, W = data.shape
    logger.info(
        f'数据形状: {data.shape}, '
        f'有效像素: {valid_mask.sum()}/{data.size} '
        f'({100 * valid_mask.sum() / data.size:.1f}%)'
    )
    logger.info(f'耗时: {time.time() - t0:.1f}s')
    return data, valid_mask, T, H, W


def natural_breaks_sampling(data, valid_mask, sample_per_class=20000):
    """自然间断法采样训练数据"""
    logger.info('[2/5] 自然间断法采样训练数据...')

    natural_breaks = [
        -float('inf'), 0.001, 0.00325, 0.0065, 0.0125,
        0.025, 0.1, float('inf')
    ]
    T, H, W = data.shape
    et, eh, ew = 50, 50, 50

    region_data = data[et:T - et, eh:H - eh, ew:W - ew]
    region_valid = valid_mask[et:T - et, eh:H - eh, ew:W - ew]

    all_positions = []
    for cls_idx in range(len(natural_breaks) - 1):
        lo, hi = natural_breaks[cls_idx], natural_breaks[cls_idx + 1]
        mask = region_valid & (region_data >= lo) & (region_data < hi)
        positions = np.argwhere(mask)
        if len(positions) == 0:
            logger.info(f'  类别 {cls_idx + 1}: 无有效像素')
            continue
        if len(positions) > sample_per_class:
            rng = np.random.RandomState(cls_idx)
            indices = rng.choice(len(positions), size=sample_per_class, replace=False)
            positions = positions[indices]
        positions[:, 0] += et
        positions[:, 1] += eh
        positions[:, 2] += ew
        all_positions.append(positions)
        logger.info(f'  类别 {cls_idx + 1}: 采样 {len(positions)} 个')

    train_positions = np.vstack(all_positions)
    logger.info(f'总训练样本: {len(train_positions)}')

    # 划分训练/验证集 (8:2)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(train_positions))
    split = int(len(train_positions) * 0.8)
    train_pos = train_positions[indices[:split]]
    val_pos = train_positions[indices[split:]]
    logger.info(f'训练集: {len(train_pos)}, 验证集: {len(val_pos)}')

    return train_pos, val_pos


def build_or_load_graphs(
    train_pos, val_pos, data, valid_mask, T, H, W,
    graph_cache_dir, skip_build=False, build_only=False
):
    """构建或加载图结构"""
    train_path = os.path.join(graph_cache_dir, 'train_graphs.pkl')
    val_path = os.path.join(graph_cache_dir, 'val_graphs.pkl')
    pos_train_path = os.path.join(graph_cache_dir, 'train_positions.npy')
    pos_val_path = os.path.join(graph_cache_dir, 'val_positions.npy')

    os.makedirs(graph_cache_dir, exist_ok=True)
    np.save(pos_train_path, train_pos)
    np.save(pos_val_path, val_pos)

    if skip_build and os.path.exists(train_path) and os.path.exists(val_path):
        logger.info('[3/5] 从缓存加载图结构...')
        t_load = time.time()
        with open(train_path, 'rb') as f:
            train_graphs = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_graphs = pickle.load(f)
        logger.info(
            f'  训练图: {len(train_graphs)}, 验证图: {len(val_graphs)}, '
            f'加载耗时: {time.time() - t_load:.1f}s'
        )
        return train_graphs, val_graphs

    logger.info('[3/5] 构建训练子图...')

    from adaptive_graph.方案C_attention_graph_learning.config_c import ConfigC
    from adaptive_graph.方案C_attention_graph_learning.graph_builder_c import GraphBuilderC

    config = ConfigC()
    config.data.data_shape = (T, H, W)
    config.graph_build.candidate_pool_size = 64
    config.graph_build.num_nodes = 36
    config.graph_build.use_mlp_selector = True
    config.graph_build.mlp_hidden_dim = 32
    config.graph_build.mlp_num_layers = 2
    config.gnn_model.num_epochs = 200
    config.gnn_model.batch_size = 64
    config.gnn_model.learning_rate = 1e-3
    config.gnn_model.early_stopping_patience = 20
    config.gnn_model.gat_hidden_dim = 64
    config.gnn_model.gat_num_heads = 4
    config.gnn_model.gat_num_layers = 2
    config.seed = 42

    builder = GraphBuilderC(config, data, valid_mask)

    def build_graphs(positions, desc='构建'):
        graphs = []
        total = len(positions)
        t_start = time.time()
        for i, pos in enumerate(positions):
            g = builder.build_single(int(pos[0]), int(pos[1]), int(pos[2]))
            if g is not None:
                graphs.append(g)
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t_start
                speed = (i + 1) / max(elapsed, 0.001)
                logger.info(
                    f'  {desc}: {i + 1}/{total} '
                    f'({speed:.0f} 图/秒), 已构建: {len(graphs)}'
                )
        elapsed = time.time() - t_start
        speed = len(graphs) / max(elapsed, 0.001)
        logger.info(
            f'  {desc} 完成: {len(graphs)} 图, '
            f'耗时 {elapsed:.1f}s, 速度 {speed:.0f} 图/秒'
        )
        return graphs

    logger.info('构建训练集子图...')
    train_graphs = build_graphs(train_pos, '训练集')

    logger.info('构建验证集子图...')
    val_graphs = build_graphs(val_pos, '验证集')

    # 保存缓存
    logger.info('保存图结构到缓存...')
    t_save = time.time()
    with open(train_path, 'wb') as f:
        pickle.dump(train_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(val_path, 'wb') as f:
        pickle.dump(val_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    train_size = os.path.getsize(train_path) / 1024 ** 3
    val_size = os.path.getsize(val_path) / 1024 ** 3
    logger.info(
        f'  训练图: {train_size:.2f} GB, 验证图: {val_size:.2f} GB, '
        f'保存耗时: {time.time() - t_save:.1f}s'
    )

    if build_only:
        logger.info('--build-only 模式，跳过训练')
        sys.exit(0)

    return train_graphs, val_graphs


def train_model(train_graphs, val_graphs, output_dir, T, H, W):
    """训练模型"""
    logger.info('[4/5] 创建数据集...')

    from adaptive_graph.方案C_attention_graph_learning.config_c import ConfigC
    from adaptive_graph.方案C_attention_graph_learning.model_c import (
        GraphDatasetC, Trainer, AttentionGraphModel
    )

    config = ConfigC()
    config.data.data_shape = (T, H, W)
    config.graph_build.candidate_pool_size = 64
    config.graph_build.num_nodes = 36
    config.graph_build.use_mlp_selector = True
    config.graph_build.mlp_hidden_dim = 32
    config.graph_build.mlp_num_layers = 2
    config.gnn_model.num_epochs = 200
    config.gnn_model.batch_size = 64
    config.gnn_model.learning_rate = 1e-3
    config.gnn_model.early_stopping_patience = 20
    config.gnn_model.gat_hidden_dim = 64
    config.gnn_model.gat_num_heads = 4
    config.gnn_model.gat_num_layers = 2
    config.output_dir = output_dir
    config.model_save_dir = os.path.join(output_dir, 'checkpoints')
    config.seed = 42

    device = config.get_device()
    logger.info(f'使用设备: {device}')

    train_dataset = GraphDatasetC(
        train_graphs, max_candidates=config.graph_build.candidate_pool_size
    )
    val_dataset = GraphDatasetC(
        val_graphs, max_candidates=config.graph_build.candidate_pool_size
    )
    logger.info(f'训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}')

    logger.info('[5/5] 训练模型...')
    model = AttentionGraphModel(config).to(device)
    trainer = Trainer(model, config)
    history = trainer.fit(train_dataset, val_dataset)

    # 保存结果
    logger.info('保存训练结果...')

    history_path = os.path.join(output_dir, 'training_history.npy')
    np.save(history_path, history)
    logger.info(f'训练历史已保存: {history_path}')

    best_loss = min(history['val_loss'])
    best_mae = history['val_mae'][np.argmin(history['val_loss'])]
    final_loss = history['train_loss'][-1]
    final_mae = history['train_mae'][-1]

    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('方案C 训练结果摘要\n')
        f.write('=' * 40 + '\n\n')
        f.write(f'训练样本: {len(train_graphs)}\n')
        f.write(f'验证样本: {len(val_graphs)}\n')
        f.write(f'设备: {device}\n\n')
        f.write(f'最佳验证 Loss: {best_loss:.6f}\n')
        f.write(f'最佳验证 MAE:  {best_mae:.4f}\n')
        f.write(f'最终训练 Loss: {final_loss:.6f}\n')
        f.write(f'最终训练 MAE:  {final_mae:.4f}\n')
        f.write(f'训练轮数: {len(history["train_loss"])}\n\n')
        f.write('逐轮结果:\n')
        f.write(f'{"Epoch":>6} {"Train_Loss":>12} {"Val_Loss":>12} '
                f'{"Train_MAE":>10} {"Val_MAE":>10}\n')
        for i in range(len(history['train_loss'])):
            f.write(
                f'{i + 1:>6} {history["train_loss"][i]:>12.6f} '
                f'{history["val_loss"][i]:>12.6f} '
                f'{history["train_mae"][i]:>10.4f} '
                f'{history["val_mae"][i]:>10.4f}\n'
            )

    logger.info(f'训练摘要已保存: {summary_path}')
    logger.info('=' * 60)
    logger.info(
        f'训练完成! 最佳验证 Loss: {best_loss:.6f}, MAE: {best_mae:.4f}'
    )
    logger.info('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='方案C 完整训练流程')
    parser.add_argument('--input', type=str, required=True, help='NTL 数据路径 (.npy)')
    parser.add_argument('--quality', type=str, required=True, help='质量标志路径 (.npy)')
    parser.add_argument('--output', type=str, default='./scheme_c_train', help='输出目录')
    parser.add_argument('--skip-build', action='store_true', help='跳过图生成，从缓存加载')
    parser.add_argument('--build-only', action='store_true', help='只生成图，不训练')
    parser.add_argument('--sample-per-class', type=int, default=20000, help='每类采样数')
    args = parser.parse_args()

    graph_cache_dir = os.path.join(args.output, 'graphs_cache')

    logger.info('=' * 60)
    logger.info('方案C 完整训练流程')
    logger.info('=' * 60)
    logger.info(f'数据: {args.input}')
    logger.info(f'质量: {args.quality}')
    logger.info(f'输出: {args.output}')
    logger.info(f'图缓存: {graph_cache_dir}')
    logger.info(f'skip_build={args.skip_build}, build_only={args.build_only}')

    # 阶段 1: 数据准备
    data, valid_mask, T, H, W = load_and_preprocess(args.input, args.quality)

    # 阶段 2: 采样
    train_pos, val_pos = natural_breaks_sampling(data, valid_mask, args.sample_per_class)

    # 阶段 3: 构建或加载图
    train_graphs, val_graphs = build_or_load_graphs(
        train_pos, val_pos, data, valid_mask, T, H, W,
        graph_cache_dir, skip_build=args.skip_build, build_only=args.build_only
    )

    # 阶段 4+5: 训练
    train_model(train_graphs, val_graphs, args.output, T, H, W)


if __name__ == '__main__':
    main()
