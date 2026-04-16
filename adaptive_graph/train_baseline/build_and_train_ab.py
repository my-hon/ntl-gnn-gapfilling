#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方案A/B 图构建 + 训练一键脚本
==============================
完成从数据加载 → 自然断点采样 → 图构建 → 保存 → 训练的全流程。

用法:
  python build_and_train_ab.py --scheme scheme_a --input /path/to/ntl.npy --quality /path/to/quality.npy --output /path/to/output
  python build_and_train_ab.py --scheme scheme_b --input /path/to/ntl.npy --quality /path/to/quality.npy --output /path/to/output
  python build_and_train_ab.py --scheme v2 --input /path/to/ntl.npy --quality /path/to/quality.npy --output /path/to/output
  python build_and_train_ab.py --skip-build --graphs /path/to/graphs.pkl --scheme scheme_a --output /path/to/output
"""

import sys
import os
import time
import pickle
import logging
import argparse
import numpy as np

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger('build_and_train_ab')


def load_and_preprocess(input_path, quality_path, edge_time=50, edge_height=50, edge_width=50):
    """加载并预处理数据（复用 v2 的 NTLDataLoader 逻辑）"""
    from ntl_graph_accel_v2.data_loader import NTLDataLoader
    from ntl_graph_accel_v2.config import Config

    config = Config()
    config.data.edge_time = edge_time
    config.data.edge_height = edge_height
    config.data.edge_width = edge_width

    loader = NTLDataLoader(config)
    loader.load(path=input_path, quality_path=quality_path if quality_path else None)
    return loader


def build_graphs(scheme, loader, sample_per_class=20000, search_node=32, output_dir='./'):
    """
    根据方案构建图结构。

    Parameters
    ----------
    scheme : str
        'v2', 'scheme_a', 'scheme_b'
    loader : NTLDataLoader
        已加载的数据
    sample_per_class : int
        每类别采样数
    search_node : int
        基础邻居节点数
    output_dir : str
        图结构保存目录

    Returns
    -------
    all_graphs : list
        所有构建好的图
    """
    # 自然断点采样
    logger.info('=' * 60)
    logger.info(f'自然断点采样 (每类 {sample_per_class} 个)...')
    sample_results = loader.get_natural_breaks_samples()

    # 合并所有类别的采样位置
    all_positions = []
    for class_idx, positions in sample_results:
        if len(positions) > 0:
            all_positions.append(positions)
    all_positions = np.concatenate(all_positions, axis=0)
    logger.info(f'总采样位置: {len(all_positions)}')

    # 创建图构建器
    if scheme == 'v2':
        from ntl_graph_accel_v2.config import Config as V2Config
        from ntl_graph_accel_v2.graph_builder import GraphBuilder

        config = V2Config()
        config.data.data_shape = loader.data.shape
        config.graph.search_node = search_node
        config.graph.sample_per_class = sample_per_class
        builder = GraphBuilder(config, loader.data)

        logger.info(f'[v2] 开始构建图 (search_node={search_node})...')
        t0 = time.time()
        all_graphs = []
        for i, (tc, hc, wc) in enumerate(all_positions):
            g = builder.build_single(int(tc), int(hc), int(wc))
            if g is not None:
                all_graphs.append(g)
            if (i + 1) % 10000 == 0:
                logger.info(f'  进度: {i+1}/{len(all_positions)}, 已构建: {len(all_graphs)}')
        logger.info(f'[v2] 图构建完成: {len(all_graphs)} 个图, 耗时 {time.time()-t0:.1f}s')

    elif scheme == 'scheme_a':
        from adaptive_graph.scheme_a_quality_adaptive_nodes.config_a import ConfigA
        from adaptive_graph.scheme_a_quality_adaptive_nodes.graph_builder_a import GraphBuilderA

        config = ConfigA()
        config.data.data_shape = loader.data.shape
        config.graph.num_nodes_base = search_node
        config.graph.sample_per_class = sample_per_class
        builder = GraphBuilderA(config, loader.data, loader.valid_mask)

        logger.info(f'[方案A] 预计算质量图...')
        builder.precompute_quality()

        logger.info(f'[方案A] 开始构建图 (base_nodes={search_node})...')
        t0 = time.time()
        all_graphs = builder.build_batch(all_positions, precompute_quality=False)
        logger.info(f'[方案A] 图构建完成: {len(all_graphs)} 个图, 耗时 {time.time()-t0:.1f}s')

        # 打印统计
        stats = builder.get_stats()
        logger.info(f'[方案A] 节点数分布: {stats["adaptive_counts"]}')

    elif scheme == 'scheme_b':
        from adaptive_graph.scheme_b_dynamic_edge_construction.config_b import ConfigB
        from adaptive_graph.scheme_b_dynamic_edge_construction.graph_builder_b import GraphBuilderB

        config = ConfigB()
        config.data.data_shape = loader.data.shape
        config.graph.search_node = search_node
        config.graph.sample_per_class = sample_per_class
        builder = GraphBuilderB(config, loader.data, loader.valid_mask)

        logger.info(f'[方案B] 开始构建图 (search_node={search_node})...')
        t0 = time.time()
        all_graphs = builder.build_batch(all_positions)
        logger.info(f'[方案B] 图构建完成: {len(all_graphs)} 个图, 耗时 {time.time()-t0:.1f}s')

    else:
        raise ValueError(f'未知方案: {scheme}')

    return all_graphs


def save_graphs(graphs, output_dir, scheme):
    """保存图结构到 pkl 文件"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{scheme}_graphs.pkl')
    t0 = time.time()
    with open(path, 'wb') as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(f'图结构已保存: {path} ({size_mb:.1f} MB, {len(graphs)} 个图, 耗时 {time.time()-t0:.1f}s)')
    return path


def train(graphs_path, scheme, output_dir, **train_kwargs):
    """调用 train_baseline.py 进行训练"""
    import importlib

    # 动态导入 train_baseline 模块
    train_baseline_dir = os.path.join(_PROJECT_ROOT, 'adaptive_graph', 'train_baseline')
    if train_baseline_dir not in sys.path:
        sys.path.insert(0, train_baseline_dir)

    # 直接调用训练函数
    from train_baseline import (
        GraphDataset, GATv2Model, train_epoch, validate,
        plot_curves, r2_score, normalize_graph,
    )
    from collections import defaultdict
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载图
    logger.info(f'加载图结构: {graphs_path}')
    t0 = time.time()
    with open(graphs_path, 'rb') as f:
        all_graphs = pickle.load(f)
    logger.info(f'  加载 {len(all_graphs)} 个图, 耗时 {time.time()-t0:.1f}s')

    # 训练参数
    hidden_dim = train_kwargs.get('hidden_dim', 128)
    num_heads = train_kwargs.get('num_heads', 8)
    num_layers = train_kwargs.get('num_layers', 4)
    batch_size = train_kwargs.get('batch_size', 256)
    epochs = train_kwargs.get('epochs', 300)
    lr = train_kwargs.get('lr', 1e-3)
    patience = train_kwargs.get('patience', 40)
    split = train_kwargs.get('split', 0.8)

    # 划分训练/验证集
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_graphs))
    split_idx = int(len(all_graphs) * split)
    train_graphs = [all_graphs[i] for i in indices[:split_idx]]
    val_graphs = [all_graphs[i] for i in indices[split_idx:]]
    logger.info(f'训练集: {len(train_graphs)}, 验证集: {len(val_graphs)}')

    # 创建 PyG Dataset
    train_dataset = GraphDataset(train_graphs)
    val_dataset = GraphDataset(val_graphs)
    logger.info(f'有效训练图: {len(train_dataset)}, 有效验证图: {len(val_dataset)}')

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 模型 - 使用更大的模型配置以提高 R²
    model = GATv2Model(
        in_dim=1,
        hidden_dim=hidden_dim,
        out_dim=1,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.15,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'模型参数量: {total_params:,}')
    logger.info(f'配置: hidden={hidden_dim}, heads={num_heads}, layers={num_layers}, batch={batch_size}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 训练
    logger.info(f'开始训练: scheme={scheme}, epochs={epochs}, device={device}')
    history = defaultdict(list)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_mae'].append(train_metrics['mae'])
        history['val_mae'].append(val_metrics['mae'])
        history['train_r2'].append(train_metrics['r2'])
        history['val_r2'].append(val_metrics['r2'])
        history['lr'].append(lr_now)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} ({time.time()-t0:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.6f} R²: {train_metrics['r2']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.6f} R²: {val_metrics['r2']:.4f} | "
                f"LR: {lr_now:.2e}"
            )

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            best_r2 = val_metrics['r2']
            torch.save(model.state_dict(), os.path.join(output_dir, f'{scheme}_best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'早停触发: {epoch+1} 轮无改善')
                break

    torch.save(model.state_dict(), os.path.join(output_dir, f'{scheme}_final.pt'))

    # 绘图
    plot_curves(history, output_dir, scheme)

    # 保存摘要
    best_idx = int(np.argmin(history['val_loss']))
    summary_path = os.path.join(output_dir, f'{scheme}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'{scheme} 训练结果\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'图数量: train={len(train_dataset)}, val={len(val_dataset)}\n')
        f.write(f'模型: hidden={hidden_dim}, heads={num_heads}, layers={num_layers}\n')
        f.write(f'Batch: {batch_size}, LR: {lr}\n\n')
        f.write(f'最佳结果 (Epoch {best_idx+1}):\n')
        f.write(f'  Val Loss: {history["val_loss"][best_idx]:.6f}\n')
        f.write(f'  Val MAE:  {history["val_mae"][best_idx]:.4f}\n')
        f.write(f'  Val R²:   {history["val_r2"][best_idx]:.4f}\n\n')
        f.write(f'逐轮结果:\n')
        f.write(f'{"Epoch":>6} {"Tr_Loss":>10} {"Va_Loss":>10} {"Tr_MAE":>8} {"Va_MAE":>8} {"Tr_R2":>8} {"Va_R2":>8}\n')
        for i in range(len(history['train_loss'])):
            f.write(f'{i+1:>6} {history["train_loss"][i]:>10.6f} {history["val_loss"][i]:>10.6f} '
                    f'{history["train_mae"][i]:>8.4f} {history["val_mae"][i]:>8.4f} '
                    f'{history["train_r2"][i]:>8.4f} {history["val_r2"][i]:>8.4f}\n')

    logger.info('=' * 60)
    logger.info(f'{scheme} 训练完成! 最佳 Val R²: {best_r2:.4f}')
    logger.info(f'摘要: {summary_path}')
    logger.info('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='方案A/B 图构建 + 训练')
    parser.add_argument('--scheme', type=str, required=True,
                        choices=['v2', 'scheme_a', 'scheme_b'],
                        help='图构建方案')
    parser.add_argument('--input', type=str, default='',
                        help='NTL 数据路径 (.npy)')
    parser.add_argument('--quality', type=str, default='',
                        help='质量标志数据路径 (.npy)')
    parser.add_argument('--output', type=str, default='./ab_results',
                        help='输出目录')
    parser.add_argument('--graphs', type=str, default='',
                        help='已有的图结构 pkl 文件（跳过构建阶段）')
    parser.add_argument('--skip-build', action='store_true',
                        help='跳过图构建，直接训练')
    parser.add_argument('--build-only', action='store_true',
                        help='只构建图，不训练')

    # 图构建参数
    parser.add_argument('--sample-per-class', type=int, default=20000,
                        help='每类别采样数')
    parser.add_argument('--search-node', type=int, default=32,
                        help='基础邻居节点数')

    # 训练参数
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='GAT 隐藏维度')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='GAT 层数')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--patience', type=int, default=40,
                        help='早停耐心')
    parser.add_argument('--split', type=float, default=0.8,
                        help='训练集比例')

    args = parser.parse_args()

    scheme_name = {
        'v2': 'v2基线',
        'scheme_a': '方案A(质量自适应)',
        'scheme_b': '方案B(异质性感知)',
    }.get(args.scheme, args.scheme)

    logger.info('=' * 60)
    logger.info(f'  {scheme_name} 图构建 + 训练')
    logger.info('=' * 60)

    graphs_dir = os.path.join(args.output, 'graphs')
    train_dir = os.path.join(args.output, 'train')

    # ---- 阶段1: 图构建 ----
    if not args.skip_build:
        if not args.input:
            parser.error('需要 --input 参数来加载数据（或使用 --skip-build 跳过构建）')

        logger.info(f'\n[阶段1] 图构建')
        logger.info(f'  数据: {args.input}')
        logger.info(f'  质量: {args.quality or "(无)"}')
        logger.info(f'  采样: {args.sample_per_class}/类')
        logger.info(f'  节点: {args.search_node}')

        # 加载数据
        loader = load_and_preprocess(args.input, args.quality)

        # 构建图
        all_graphs = build_graphs(
            args.scheme, loader,
            sample_per_class=args.sample_per_class,
            search_node=args.search_node,
            output_dir=graphs_dir,
        )

        # 保存图
        graphs_path = save_graphs(all_graphs, graphs_dir, args.scheme)
    else:
        if not args.graphs:
            parser.error('--skip-build 需要 --graphs 参数指定已有的图结构文件')
        graphs_path = args.graphs
        logger.info(f'\n[阶段1] 跳过图构建，使用已有文件: {graphs_path}')

    # ---- 阶段2: 训练 ----
    if not args.build_only:
        logger.info(f'\n[阶段2] GNN 训练')
        logger.info(f'  图文件: {graphs_path}')
        logger.info(f'  模型: hidden={args.hidden_dim}, heads={args.num_heads}, layers={args.num_layers}')
        logger.info(f'  训练: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}')

        train_kwargs = {
            'hidden_dim': args.hidden_dim,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'patience': args.patience,
            'split': args.split,
        }

        train(graphs_path, args.scheme, train_dir, **train_kwargs)
    else:
        logger.info(f'\n[阶段2] 跳过训练 (--build-only)')

    logger.info(f'\n全部完成! 输出目录: {args.output}')


if __name__ == '__main__':
    main()
