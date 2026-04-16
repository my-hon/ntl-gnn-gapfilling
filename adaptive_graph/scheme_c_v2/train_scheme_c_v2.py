#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方案C v2 增强训练脚本

相比 v1 的改进:
1. 增加 R² 指标（训练集和验证集）
2. 训练结束后自动绘制损失曲线（Loss、MAE、R²、LR）
3. 增大 batch_size（64→256）提升 GPU 利用率
4. 增加 GNN 层数（2→4）和隐藏维度（64→128）
5. 增加 num_workers（4→8）加速数据加载

用法:
  python train_scheme_c_v2.py --skip-build
  python train_scheme_c_v2.py --build-only
  python train_scheme_c_v2.py                    # 完整流程
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

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from adaptive_graph.scheme_c_attention_graph_learning.config_c import ConfigC
from adaptive_graph.scheme_c_attention_graph_learning.graph_builder_c import GraphBuilderC
from adaptive_graph.scheme_c_attention_graph_learning.model_c import (
    GraphDatasetC, collate_fn_c, AttentionGraphModel
)

# ================================================================
# 日志配置
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger('scheme_c_v2')


# ================================================================
# R² 计算
# ================================================================
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 R² (决定系数)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ================================================================
# 增强版 Trainer（继承原始 Trainer 的逻辑，添加 R² 和绘图）
# ================================================================
class EnhancedTrainer:
    """增强版训练器：添加 R² 指标、更大 batch、更多 GNN 层"""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.gnn_cfg = config.gnn_model

        # 优化器
        gnn_params = [
            p for n, p in model.named_parameters()
            if 'graph_learner' not in n and p.requires_grad
        ]
        graph_params = [
            p for n, p in model.named_parameters()
            if 'graph_learner' in n and p.requires_grad
        ]

        self.optimizer_gnn = torch.optim.Adam(
            gnn_params, lr=self.gnn_cfg.learning_rate, weight_decay=1e-5
        )
        self.optimizer_graph = torch.optim.Adam(
            graph_params, lr=self.gnn_cfg.learning_rate * self.gnn_cfg.graph_lr_ratio
        )

        # 学习率调度器
        self.scheduler_gnn = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_gnn, T_max=self.gnn_cfg.num_epochs, eta_min=1e-6
        )
        self.scheduler_graph = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_graph, T_max=self.gnn_cfg.num_epochs, eta_min=1e-7
        )

        # 损失函数
        self.criterion = torch.nn.MSELoss()
        self.mae_criterion = torch.nn.L1Loss()

        # 状态
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0
        self.history = defaultdict(list)

    def train_epoch(self, dataloader):
        """训练一个 epoch，返回指标字典"""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_reg = 0.0
        all_preds = []
        all_targets = []
        num_batches = 0

        for batch in dataloader:
            candidate_features = batch["candidate_features"].to(self.device)
            candidate_offsets = batch["candidate_offsets"].to(self.device)
            candidate_mask = batch["candidate_mask"].to(self.device)
            targets = batch["targets"].to(self.device)

            result = self.model(candidate_features, candidate_offsets, candidate_mask)
            prediction = result["prediction"]
            reg_loss = result["reg_loss"]

            pred_loss = self.criterion(prediction, targets)
            mae_loss = self.mae_criterion(prediction, targets)
            loss = pred_loss + reg_loss

            self.optimizer_gnn.zero_grad()
            self.optimizer_graph.zero_grad()
            loss.backward()

            if self.gnn_cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gnn_cfg.gradient_clip
                )

            self.optimizer_gnn.step()
            self.optimizer_graph.step()

            total_loss += pred_loss.item()
            total_mae += mae_loss.item()
            total_reg += reg_loss.item()
            num_batches += 1

            all_preds.append(prediction.detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0).flatten()
        tgts = np.concatenate(all_targets, axis=0).flatten()
        r2 = r2_score(tgts, preds)

        return {
            "loss": total_loss / max(num_batches, 1),
            "mae": total_mae / max(num_batches, 1),
            "reg_loss": total_reg / max(num_batches, 1),
            "r2": r2,
        }

    @torch.no_grad()
    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        all_preds = []
        all_targets = []
        num_batches = 0

        for batch in dataloader:
            candidate_features = batch["candidate_features"].to(self.device)
            candidate_offsets = batch["candidate_offsets"].to(self.device)
            candidate_mask = batch["candidate_mask"].to(self.device)
            targets = batch["targets"].to(self.device)

            result = self.model(candidate_features, candidate_offsets, candidate_mask)
            prediction = result["prediction"]

            pred_loss = self.criterion(prediction, targets)
            mae_loss = self.mae_criterion(prediction, targets)

            total_loss += pred_loss.item()
            total_mae += mae_loss.item()
            num_batches += 1

            all_preds.append(prediction.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0).flatten()
        tgts = np.concatenate(all_targets, axis=0).flatten()
        r2 = r2_score(tgts, preds)

        return {
            "loss": total_loss / max(num_batches, 1),
            "mae": total_mae / max(num_batches, 1),
            "r2": r2,
        }

    def save_checkpoint(self, filename, epoch):
        path = os.path.join(self.config.model_save_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_gnn_state_dict": self.optimizer_gnn.state_dict(),
            "optimizer_graph_state_dict": self.optimizer_graph.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": dict(self.history),
        }, path)

    def fit(self, train_dataset, val_dataset):
        """完整训练流程"""
        from torch.utils.data import DataLoader

        gnn_cfg = self.gnn_cfg

        train_loader = DataLoader(
            train_dataset,
            batch_size=gnn_cfg.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn_c,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=gnn_cfg.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=collate_fn_c,
                pin_memory=True,
            )

        logger.info(
            f"开始训练: epochs={gnn_cfg.num_epochs}, "
            f"batch_size={gnn_cfg.batch_size}, "
            f"gat_layers={gnn_cfg.gat_num_layers}, "
            f"gat_hidden={gnn_cfg.gat_hidden_dim}, "
            f"gat_heads={gnn_cfg.gat_num_heads}, "
            f"device={self.device}"
        )
        logger.info(
            f"训练集: {len(train_dataset)} 样本, "
            f"验证集: {len(val_dataset) if val_dataset else 0} 样本"
        )

        for epoch in range(gnn_cfg.num_epochs):
            self.epoch = epoch
            t0 = time.time()

            train_metrics = self.train_epoch(train_loader)

            val_metrics = {"loss": float('inf'), "mae": float('inf'), "r2": 0.0}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)

            self.scheduler_gnn.step()
            self.scheduler_graph.step()

            current_lr = self.optimizer_gnn.param_groups[0]['lr']
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_mae"].append(train_metrics["mae"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["train_r2"].append(train_metrics["r2"])
            self.history["val_r2"].append(val_metrics["r2"])
            self.history["reg_loss"].append(train_metrics["reg_loss"])
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{gnn_cfg.num_epochs} "
                    f"({elapsed:.1f}s) | "
                    f"Train Loss: {train_metrics['loss']:.6f} "
                    f"MAE: {train_metrics['mae']:.4f} "
                    f"R²: {train_metrics['r2']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.6f} "
                    f"MAE: {val_metrics['mae']:.4f} "
                    f"R²: {val_metrics['r2']:.4f} | "
                    f"Reg: {train_metrics['reg_loss']:.6f} | "
                    f"LR: {current_lr:.2e}"
                )

            if val_loader is not None:
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0
                    self.save_checkpoint("best_model.pt", epoch)
                    logger.info(
                        f"  ★ 最佳模型更新: Val Loss={val_metrics['loss']:.6f}, "
                        f"R²={val_metrics['r2']:.4f}"
                    )
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= gnn_cfg.early_stopping_patience:
                        logger.info(f"早停触发: {epoch+1} 轮无改善")
                        break

        self.save_checkpoint("final_model.pt", self.epoch)
        logger.info(f"训练完成! 最佳验证损失: {self.best_val_loss:.6f}")

        return dict(self.history)


# ================================================================
# 绘图
# ================================================================
def plot_training_curves(history, output_dir):
    """绘制训练曲线并保存"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss 曲线
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. MAE 曲线
    ax = axes[0, 1]
    ax.plot(epochs, history['train_mae'], 'b-', label='Train MAE', linewidth=1.5)
    ax.plot(epochs, history['val_mae'], 'r-', label='Val MAE', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Training & Validation MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. R² 曲线
    ax = axes[1, 0]
    ax.plot(epochs, history['train_r2'], 'b-', label='Train R²', linewidth=1.5)
    ax.plot(epochs, history['val_r2'], 'r-', label='Val R²', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R²')
    ax.set_title('Training & Validation R²')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 4. Learning Rate 曲线
    ax = axes[1, 1]
    ax.plot(epochs, history['lr'], 'g-', label='Learning Rate', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"训练曲线已保存: {plot_path}")


# ================================================================
# 数据加载与预处理
# ================================================================
def load_and_preprocess(ntl_path, quality_path):
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

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(train_positions))
    split = int(len(train_positions) * 0.8)
    train_pos = train_positions[indices[:split]]
    val_pos = train_positions[indices[split:]]
    logger.info(f'训练集: {len(train_pos)}, 验证集: {len(val_pos)}')

    return train_pos, val_pos


# ================================================================
# 图构建（带缓存）
# ================================================================
def build_or_load_graphs(
    train_pos, val_pos, data, valid_mask, T, H, W,
    graph_cache_dir, skip_build=False, build_only=False
):
    train_path = os.path.join(graph_cache_dir, 'train_graphs.pkl')
    val_path = os.path.join(graph_cache_dir, 'val_graphs.pkl')

    os.makedirs(graph_cache_dir, exist_ok=True)
    np.save(os.path.join(graph_cache_dir, 'train_positions.npy'), train_pos)
    np.save(os.path.join(graph_cache_dir, 'val_positions.npy'), val_pos)

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

    config = ConfigC()
    config.data.data_shape = (T, H, W)
    config.graph_build.candidate_pool_size = 64
    config.graph_build.num_nodes = 36
    config.graph_build.use_mlp_selector = True
    config.graph_build.mlp_hidden_dim = 32
    config.graph_build.mlp_num_layers = 2
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


# ================================================================
# 训练（v2 增强）
# ================================================================
def train_model(train_graphs, val_graphs, output_dir, T, H, W):
    logger.info('[4/5] 创建数据集...')

    config = ConfigC()
    config.data.data_shape = (T, H, W)
    config.graph_build.candidate_pool_size = 64
    config.graph_build.num_nodes = 36
    config.graph_build.use_mlp_selector = True
    config.graph_build.mlp_hidden_dim = 32
    config.graph_build.mlp_num_layers = 2

    # v2 增强: 更大模型和 batch
    config.gnn_model.num_epochs = 200
    config.gnn_model.batch_size = 256          # 64 → 256
    config.gnn_model.learning_rate = 1e-3
    config.gnn_model.early_stopping_patience = 30  # 20 → 30
    config.gnn_model.gat_hidden_dim = 128      # 64 → 128
    config.gnn_model.gat_num_heads = 8          # 4 → 8
    config.gnn_model.gat_num_layers = 4         # 2 → 4
    config.gnn_model.gat_dropout = 0.15         # 0.1 → 0.15
    config.gnn_model.gradient_clip = 1.0
    config.gnn_model.joint_training = True
    config.gnn_model.graph_lr_ratio = 0.1
    config.gnn_model.freeze_graph_after_epoch = 150

    config.output_dir = output_dir
    config.model_save_dir = os.path.join(output_dir, 'checkpoints')
    config.num_workers = 8                     # 4 → 8
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

    logger.info('[5/5] 训练模型 (v2 增强)...')
    model = AttentionGraphModel(config).to(device)
    trainer = EnhancedTrainer(model, config, device)

    history = trainer.fit(train_dataset, val_dataset)

    # 绘制训练曲线
    logger.info('绘制训练曲线...')
    plot_training_curves(history, output_dir)

    # 保存训练历史
    history_path = os.path.join(output_dir, 'training_history_v2.npy')
    np.save(history_path, history)
    logger.info(f'训练历史已保存: {history_path}')

    # 保存摘要
    best_idx = int(np.argmin(history['val_loss']))
    summary_path = os.path.join(output_dir, 'training_summary_v2.txt')
    with open(summary_path, 'w') as f:
        f.write('方案C v2 增强训练结果摘要\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'训练样本: {len(train_graphs)}\n')
        f.write(f'验证样本: {len(val_graphs)}\n')
        f.write(f'设备: {device}\n\n')
        f.write(f'模型配置:\n')
        f.write(f'  GAT layers: {config.gnn_model.gat_num_layers}\n')
        f.write(f'  GAT hidden: {config.gnn_model.gat_hidden_dim}\n')
        f.write(f'  GAT heads: {config.gnn_model.gat_num_heads}\n')
        f.write(f'  Batch size: {config.gnn_model.batch_size}\n')
        f.write(f'  Learning rate: {config.gnn_model.learning_rate}\n\n')
        f.write(f'最佳结果 (Epoch {best_idx + 1}):\n')
        f.write(f'  Val Loss: {history["val_loss"][best_idx]:.6f}\n')
        f.write(f'  Val MAE:  {history["val_mae"][best_idx]:.4f}\n')
        f.write(f'  Val R²:   {history["val_r2"][best_idx]:.4f}\n\n')
        f.write(f'最终结果 (Epoch {len(history["train_loss"])}):\n')
        f.write(f'  Train Loss: {history["train_loss"][-1]:.6f}\n')
        f.write(f'  Train MAE:  {history["train_mae"][-1]:.4f}\n')
        f.write(f'  Train R²:   {history["train_r2"][-1]:.4f}\n')
        f.write(f'  Val Loss:   {history["val_loss"][-1]:.6f}\n')
        f.write(f'  Val MAE:    {history["val_mae"][-1]:.4f}\n')
        f.write(f'  Val R²:     {history["val_r2"][-1]:.4f}\n\n')
        f.write('逐轮结果:\n')
        f.write(f'{"Epoch":>6} {"Tr_Loss":>10} {"Va_Loss":>10} '
                f'{"Tr_MAE":>8} {"Va_MAE":>8} '
                f'{"Tr_R2":>8} {"Va_R2":>8} {"LR":>10}\n')
        for i in range(len(history['train_loss'])):
            f.write(
                f'{i + 1:>6} {history["train_loss"][i]:>10.6f} '
                f'{history["val_loss"][i]:>10.6f} '
                f'{history["train_mae"][i]:>8.4f} {history["val_mae"][i]:>8.4f} '
                f'{history["train_r2"][i]:>8.4f} {history["val_r2"][i]:>8.4f} '
                f'{history["lr"][i]:>10.2e}\n'
            )

    logger.info(f'训练摘要已保存: {summary_path}')
    logger.info('=' * 60)
    logger.info(
        f'训练完成! 最佳 Val Loss: {history["val_loss"][best_idx]:.6f}, '
        f'R²: {history["val_r2"][best_idx]:.4f}'
    )
    logger.info('=' * 60)


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='方案C v2 增强训练')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--quality', type=str, required=True)
    parser.add_argument('--output', type=str, default='./scheme_c_v2_train')
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--sample-per-class', type=int, default=20000)
    args = parser.parse_args()

    graph_cache_dir = os.path.join(args.output, 'graphs_cache')

    logger.info('=' * 60)
    logger.info('方案C v2 增强训练流程')
    logger.info('=' * 60)
    logger.info(f'数据: {args.input}')
    logger.info(f'质量: {args.quality}')
    logger.info(f'输出: {args.output}')
    logger.info(f'图缓存: {graph_cache_dir}')
    logger.info(f'skip_build={args.skip_build}, build_only={args.build_only}')

    data, valid_mask, T, H, W = load_and_preprocess(args.input, args.quality)
    train_pos, val_pos = natural_breaks_sampling(data, valid_mask, args.sample_per_class)
    train_graphs, val_graphs = build_or_load_graphs(
        train_pos, val_pos, data, valid_mask, T, H, W,
        graph_cache_dir, skip_build=args.skip_build, build_only=args.build_only
    )
    train_model(train_graphs, val_graphs, args.output, T, H, W)


if __name__ == '__main__':
    main()
