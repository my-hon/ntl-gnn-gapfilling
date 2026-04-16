#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基线 GNN 训练脚本（适用于 v2 / 方案A / 方案B 的图结构）

使用 PyG 的 GATv2 进行节点级回归（预测中心节点值）。

用法:
  python train_baseline.py --graphs /path/to/graphs.pkl --scheme v2
  python train_baseline.py --graphs /path/to/graphs.pkl --scheme scheme_a
  python train_baseline.py --graphs /path/to/graphs.pkl --scheme scheme_b
"""

import sys
import os
import time
import logging
import pickle
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger('train_baseline')


# ================================================================
# 图格式转换
# ================================================================
def normalize_graph(graph):
    """
    将各种格式的图统一转换为 PyG Data 对象。
    支持格式: v2 dict, scheme_a SubGraph, scheme_b SubGraph
    """
    # dict 格式 (v2 / scheme_a to_dict)
    if isinstance(graph, dict):
        nf = graph['node_features']  # (N, 1)
        ei = graph['edge_index']     # (2, E)
        ea = graph['edge_attr']      # (E, 3)
        gt = graph['ground_truth']   # (N, 1)
        pos = graph.get('position', np.zeros((1, 3)))
    else:
        # SubGraph dataclass
        nf = graph.node_features
        gt = graph.ground_truth if hasattr(graph, 'ground_truth') else None
        if gt is None:
            # scheme_b: 用 center_value 构造 ground_truth
            cv = graph.center_value if hasattr(graph, 'center_value') else 0.0
            gt = np.full((nf.shape[0], 1), cv, dtype=np.float32)
            gt[0] = cv

        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            ei = graph.edge_index
        elif hasattr(graph, 'edge_index_src'):
            ei = np.stack([graph.edge_index_src, graph.edge_index_dst], axis=0)
        else:
            ei = np.zeros((2, 0), dtype=np.int64)

        ea = graph.edge_attr if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
        if ea is None and hasattr(graph, 'edge_attrs') and graph.edge_attrs is not None:
            ea = graph.edge_attrs

        pos = graph.position if hasattr(graph, 'position') and graph.position is not None else None
        if pos is None and hasattr(graph, 'center_pos'):
            pos = graph.center_pos.reshape(1, 3)

    # 确保 dtype 正确
    x = torch.from_numpy(np.asarray(nf, dtype=np.float32))
    edge_index = torch.from_numpy(np.asarray(ei, dtype=np.int64))
    edge_attr = torch.from_numpy(np.asarray(ea, dtype=np.float32)) if ea is not None else None
    y = torch.from_numpy(np.asarray(gt, dtype=np.float32))

    # 目标: 预测中心节点 (index=0) 的值
    target = y[0:1]  # (1, 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=target)


# ================================================================
# PyG Dataset
# ================================================================
class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.data_list = []
        for g in graphs:
            try:
                pyg_data = normalize_graph(g)
                if pyg_data.x.shape[0] > 1 and pyg_data.edge_index.shape[1] > 0:
                    self.data_list.append(pyg_data)
            except Exception as e:
                pass  # 跳过无效图

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# ================================================================
# GATv2 模型
# ================================================================
class GATv2Model(torch.nn.Module):
    def __init__(self, in_dim=1, hidden_dim=64, out_dim=1, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        # 第一层
        self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
        self.norms.append(torch.nn.LayerNorm(hidden_dim * num_heads))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
            self.norms.append(torch.nn.LayerNorm(hidden_dim * num_heads))

        # 最后一层
        self.convs.append(GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, concat=False))
        self.norms.append(torch.nn.LayerNorm(hidden_dim))

        # 读出: 中心节点嵌入 → 预测值
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index, edge_attr=edge_attr)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=0.1, training=self.training)
            x = h + (x if i == 0 and x.shape[-1] == h.shape[-1] else 0)  # 残差（维度匹配时）

        # 中心节点 (index=0) 的嵌入 → 预测
        center_emb = x[0:1]  # (1, hidden_dim)
        pred = self.readout(center_emb)  # (1, 1)
        return pred


# ================================================================
# R² 计算
# ================================================================
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ================================================================
# 训练与验证
# ================================================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    all_preds = []
    all_targets = []
    n = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)  # (B, 1)
        loss = F.mse_loss(pred, batch.y)
        mae = F.l1_loss(pred, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_mae += mae.item() * batch.num_graphs
        n += batch.num_graphs
        all_preds.append(pred.detach().cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    tgts = np.concatenate(all_targets).flatten()
    r2 = r2_score(tgts, preds)

    return {
        "loss": total_loss / max(n, 1),
        "mae": total_mae / max(n, 1),
        "r2": r2,
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    all_preds = []
    all_targets = []
    n = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = F.mse_loss(pred, batch.y)
        mae = F.l1_loss(pred, batch.y)

        total_loss += loss.item() * batch.num_graphs
        total_mae += mae.item() * batch.num_graphs
        n += batch.num_graphs
        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    tgts = np.concatenate(all_targets).flatten()
    r2 = r2_score(tgts, preds)

    return {
        "loss": total_loss / max(n, 1),
        "mae": total_mae / max(n, 1),
        "r2": r2,
    }


# ================================================================
# 绘图
# ================================================================
def plot_curves(history, output_dir, scheme_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(f'{scheme_name} - Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE
    ax = axes[1]
    ax.plot(epochs, history['train_mae'], 'b-', label='Train MAE', linewidth=1.5)
    ax.plot(epochs, history['val_mae'], 'r-', label='Val MAE', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title(f'{scheme_name} - MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R²
    ax = axes[2]
    ax.plot(epochs, history['train_r2'], 'b-', label='Train R²', linewidth=1.5)
    ax.plot(epochs, history['val_r2'], 'r-', label='Val R²', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R²')
    ax.set_title(f'{scheme_name} - R²')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Target R²=0.9')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, f'{scheme_name}_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'训练曲线已保存: {path}')


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='基线 GNN 训练')
    parser.add_argument('--graphs', type=str, required=True, help='图结构 pkl 文件路径')
    parser.add_argument('--scheme', type=str, default='v2', choices=['v2', 'scheme_a', 'scheme_b'],
                        help='图构建方案名称')
    parser.add_argument('--output', type=str, default='./baseline_results', help='输出目录')
    parser.add_argument('--hidden-dim', type=int, default=64, help='GAT 隐藏维度')
    parser.add_argument('--num-heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=3, help='GAT 层数')
    parser.add_argument('--batch-size', type=int, default=256, help='批次大小')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--patience', type=int, default=30, help='早停耐心')
    parser.add_argument('--split', type=float, default=0.8, help='训练集比例')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载图
    logger.info(f'加载图结构: {args.graphs}')
    t0 = time.time()
    with open(args.graphs, 'rb') as f:
        all_graphs = pickle.load(f)
    logger.info(f'  加载 {len(all_graphs)} 个图, 耗时 {time.time()-t0:.1f}s')

    # 划分训练/验证集
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_graphs))
    split = int(len(all_graphs) * args.split)
    train_graphs = [all_graphs[i] for i in indices[:split]]
    val_graphs = [all_graphs[i] for i in indices[split:]]
    logger.info(f'训练集: {len(train_graphs)}, 验证集: {len(val_graphs)}')

    # 创建 PyG Dataset
    logger.info('创建 PyG Dataset...')
    train_dataset = GraphDataset(train_graphs)
    val_dataset = GraphDataset(val_graphs)
    logger.info(f'有效训练图: {len(train_dataset)}, 有效验证图: {len(val_dataset)}')

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型
    model = GATv2Model(
        in_dim=1,
        hidden_dim=args.hidden_dim,
        out_dim=1,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.15,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'模型参数量: {total_params:,}')
    logger.info(f'配置: hidden={args.hidden_dim}, heads={args.num_heads}, layers={args.num_layers}, batch={args.batch_size}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练
    logger.info(f'开始训练: scheme={args.scheme}, epochs={args.epochs}, device={device}')
    history = defaultdict(list)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_mae'].append(train_metrics['mae'])
        history['val_mae'].append(val_metrics['mae'])
        history['train_r2'].append(train_metrics['r2'])
        history['val_r2'].append(val_metrics['r2'])
        history['lr'].append(lr)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} ({time.time()-t0:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.6f} R²: {train_metrics['r2']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.6f} R²: {val_metrics['r2']:.4f} | "
                f"LR: {lr:.2e}"
            )

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            best_r2 = val_metrics['r2']
            torch.save(model.state_dict(), os.path.join(args.output, f'{args.scheme}_best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f'早停触发: {epoch+1} 轮无改善')
                break

    torch.save(model.state_dict(), os.path.join(args.output, f'{args.scheme}_final.pt'))

    # 绘图
    plot_curves(history, args.output, args.scheme)

    # 保存摘要
    best_idx = int(np.argmin(history['val_loss']))
    summary_path = os.path.join(args.output, f'{args.scheme}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'{args.scheme} 训练结果\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'图数量: train={len(train_dataset)}, val={len(val_dataset)}\n')
        f.write(f'模型: hidden={args.hidden_dim}, heads={args.num_heads}, layers={args.num_layers}\n')
        f.write(f'Batch: {args.batch_size}, LR: {args.lr}\n\n')
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
    logger.info(f'{args.scheme} 训练完成! 最佳 Val R²: {best_r2:.4f}')
    logger.info(f'摘要: {summary_path}')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
