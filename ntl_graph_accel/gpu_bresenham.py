"""
GPU加速3D Bresenham算法模块
============================
提供两种加速方式：
1. CUDA Kernel加速（需要CUDA环境）
2. 预计算查找表（CPU fallback，消除重复计算）
"""

import numpy as np
import torch
import logging
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class BresenhamLookupTable:
    """
    3D Bresenham查找表。
    
    预计算所有可能偏移范围内的Bresenham路径，
    运行时直接查表，消除重复计算。
    """

    def __init__(self, max_radius: int = 20):
        self.max_radius = max_radius
        self.max_len = 3 * (2 * max_radius + 1)  # 最长路径
        self._table = {}

    def build(self):
        """构建查找表"""
        r = self.max_radius
        logger.info(f"构建Bresenham查找表, 半径范围=[-{r}, {r}]...")

        count = 0
        for dt in range(-r, r + 1):
            for dh in range(-r, r + 1):
                for dw in range(-r, r + 1):
                    if dt == 0 and dh == 0 and dw == 0:
                        continue
                    path = self._bresenham_3d_cpu(dt, dh, dw)
                    self._table[(dt, dh, dw)] = path
                    count += 1

        logger.info(f"查找表构建完成, 共{count}条路径")

    def lookup(self, dt: int, dh: int, dw: int) -> np.ndarray:
        """查询Bresenham路径"""
        return self._table.get((dt, dh, dw), np.array([], dtype=np.int32))

    def save(self, path: str):
        """保存查找表"""
        # 将key转为字符串以便序列化
        table_serializable = {
            f"{k[0]},{k[1]},{k[2]}": v.tolist()
            for k, v in self._table.items()
        }
        np.savez_compressed(path, **table_serializable)
        logger.info(f"查找表已保存至 {path}")

    def load(self, path: str):
        """加载查找表"""
        data = np.load(path, allow_pickle=True)
        self._table = {}
        for key, val in data.items():
            dt, dh, dw = map(int, key.split(','))
            self._table[(dt, dh, dw)] = np.array(val, dtype=np.int32)
        logger.info(f"查找表已从 {path} 加载, 共{len(self._table)}条路径")

    @staticmethod
    def _bresenham_3d_cpu(dt: int, dh: int, dw: int) -> np.ndarray:
        """
        CPU版3D Bresenham算法。
        返回从(0,0,0)到(dt,dh,dw)经过的所有体素索引序列（不含端点）。
        """
        path = []
        x, y, z = 0, 0, 0

        steps = max(abs(dt), abs(dh), abs(dw))
        if steps == 0:
            return np.array([], dtype=np.int32)

        # 增量
        if dt != 0:
            dx = 1 if dt > 0 else -1
            sdx = abs(dt)
        else:
            dx, sdx = 0, 1

        if dh != 0:
            dy = 1 if dh > 0 else -1
            sdy = abs(dh)
        else:
            dy, sdy = 0, 1

        if dw != 0:
            dz = 1 if dw > 0 else -1
            sdz = abs(dw)
        else:
            dz, sdz = 0, 1

        # 误差累积器
        ey = -(steps // 2)
        ez = -(steps // 2)

        for _ in range(steps):
            # 记录当前点（跳过起点和终点）
            if not (x == 0 and y == 0 and z == 0):
                path.append((x, y, z))

            # 更新坐标
            ey += sdy
            ez += sdz

            if ey >= 0:
                y += dy
                ey -= steps

            if ez >= 0:
                z += dz
                ez -= steps

            x += dx

        # 移除终点
        if path and path[-1] == (dt, dh, dw):
            path.pop()

        return np.array(path, dtype=np.int32) if path else np.array([], dtype=np.int32)


class GPUBresenhamAccelerator:
    """
    GPU加速Bresenham算法。
    
    使用PyTorch CUDA实现批量3D线段追踪，
    一次处理数千条线段。
    """

    def __init__(self, max_radius: int = 20, max_path_len: int = 60):
        self.max_radius = max_radius
        self.max_path_len = max_path_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.device.type == 'cuda':
            logger.info(f"GPU Bresenham加速器初始化, 设备: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA不可用, 将回退到CPU模式")

    def batch_compute(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        valid_mask_cube: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量计算Bresenham路径。

        Parameters
        ----------
        starts : np.ndarray
            起点坐标, 形状 (N, 3)
        ends : np.ndarray
            终点坐标, 形状 (N, 3)
        valid_mask_cube : np.ndarray, optional
            时空立方体有效掩码, 用于过滤NaN中间点

        Returns
        -------
        paths : np.ndarray
            路径序列, 形状 (N, max_path_len, 3), 无效位置填充-1
        lengths : np.ndarray
            每条路径的实际长度, 形状 (N,)
        """
        N = starts.shape[0]
        if N == 0:
            return np.zeros((0, self.max_path_len, 3), dtype=np.int32), np.zeros(0, dtype=np.int32)

        starts_t = torch.from_numpy(starts).long().to(self.device)
        ends_t = torch.from_numpy(ends).long().to(self.device)

        # 计算每条线段的步数和增量
        deltas = ends_t - starts_t
        steps = torch.max(torch.abs(deltas), dim=1).values  # (N,)

        # 预分配输出
        paths = torch.full(
            (N, self.max_path_len, 3), -1, dtype=torch.long, device=self.device
        )
        lengths = torch.zeros(N, dtype=torch.long, device=self.device)

        # 向量化Bresenham
        # 对每条线段逐步追踪
        max_steps = steps.max().item()

        # 当前位置
        current = starts_t.clone()  # (N, 3)

        # 误差累积器
        abs_deltas = torch.abs(deltas).float()
        abs_deltas = torch.where(abs_deltas == 0, torch.ones_like(abs_deltas), abs_deltas)
        directions = torch.sign(deltas).long()  # (N, 3)
        steps_float = steps.float().unsqueeze(1)  # (N, 1)

        # 初始化误差
        errors = -steps_float / 2.0  # (N, 1) 广播到3维

        for step in range(max_steps):
            # 判断哪些线段还需要继续
            active = (torch.arange(N, device=self.device) < steps) & \
                     (lengths < self.max_path_len)

            if not active.any():
                break

            # 跳过起点（step=0时current == start）
            if step > 0:
                # 记录当前位置
                valid_write = active & (lengths < self.max_path_len)
                write_idx = torch.where(valid_write)[0]
                if write_idx.numel() > 0:
                    paths[write_idx, lengths[write_idx]] = current[write_idx]
                    lengths[write_idx] += 1

            # 更新误差和坐标
            for dim in range(3):
                errors_active = errors[:, dim].clone()
                errors_active += abs_deltas[:, dim]

                # 判断是否需要在该维度步进
                need_step = (errors_active >= 0) & active
                current[need_step, dim] += directions[need_step, dim]
                errors[need_step, dim] = errors_active[need_step] - steps[need_step]

            # 第0维始终步进（主轴）
            current[active, 0] += directions[active, 0]

        # 过滤终点
        for i in range(N):
            l = lengths[i].item()
            if l > 0:
                last = paths[i, l - 1].cpu().numpy()
                end = ends[i].numpy()
                if np.array_equal(last, end):
                    lengths[i] -= 1

        return paths.cpu().numpy(), lengths.cpu().numpy()

    def batch_filter_nan(
        self,
        paths: np.ndarray,
        lengths: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        过滤路径中的NaN点。

        Parameters
        ----------
        paths : np.ndarray
            形状 (N, max_len, 3)
        lengths : np.ndarray
            形状 (N,)
        valid_mask : np.ndarray
            时空立方体有效掩码

        Returns
        -------
        filtered_paths : np.ndarray
        filtered_lengths : np.ndarray
        """
        N = paths.shape[0]
        T, H, W = valid_mask.shape

        for i in range(N):
            l = lengths[i]
            if l == 0:
                continue

            # 检查路径中的每个点
            new_l = 0
            for j in range(l):
                t, h, w = paths[i, j]
                if 0 <= t < T and 0 <= h < H and 0 <= w < W:
                    if valid_mask[t, h, w]:
                        if new_l != j:
                            paths[i, new_l] = paths[i, j]
                        new_l += 1
                    else:
                        # 遇到NaN，截断后续路径
                        break
                else:
                    break
            lengths[i] = new_l

        return paths, lengths
