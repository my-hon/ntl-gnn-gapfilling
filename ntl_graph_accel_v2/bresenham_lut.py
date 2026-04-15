"""
Bresenham 查找表模块（v2）
===========================
预计算所有偏移的 3D Bresenham 路径，存储为紧凑的 numpy 数组。
查找表可直接被 Numba JIT 函数访问（纯 numpy，无 Python 对象）。
"""

import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class BresenhamLUT:
    """
    3D Bresenham 查找表。

    存储格式（Numba 友好）：
      lut_array:  shape (2R+1, 2R+1, 2R+1, max_len, 3), dtype=int16
                 lut_array[dt+R, dh+R, dw+R] = 从(0,0,0)到(dt,dh,dw)的路径体素
      lut_lengths: shape (2R+1, 2R+1, 2R+1), dtype=int16
                 每条路径的实际长度，-1 表示无效（原点）
    """

    def __init__(self, max_radius: int = 20):
        self.max_radius = max_radius
        self.max_len = 3 * (2 * max_radius + 1)
        self.lut_array = None    # (2R+1, 2R+1, 2R+1, max_len, 3)
        self.lut_lengths = None  # (2R+1, 2R+1, 2R+1)

    def build(self):
        """构建查找表"""
        R = self.max_radius
        size = 2 * R + 1
        logger.info(f"构建 Bresenham 查找表, R={R}, size={size}x{size}x{size}...")

        self.lut_array = np.full((size, size, size, self.max_len, 3),
                                  -1, dtype=np.int16)
        self.lut_lengths = np.full((size, size, size), -1, dtype=np.int16)

        for dt in range(-R, R + 1):
            for dh in range(-R, R + 1):
                for dw in range(-R, R + 1):
                    if dt == 0 and dh == 0 and dw == 0:
                        continue
                    path = self._bresenham_3d(dt, dh, dw)
                    idx = (dt + R, dh + R, dw + R)
                    self.lut_lengths[idx] = len(path)
                    for k, (pt, ph, pw) in enumerate(path):
                        self.lut_array[idx, k] = [pt, ph, pw]

        logger.info(f"查找表构建完成, "
                     f"lut_array: {self.lut_array.nbytes / 1024 / 1024:.1f}MB")

    def save(self, path: str):
        """保存查找表"""
        np.savez_compressed(path,
                            lut_array=self.lut_array,
                            lut_lengths=self.lut_lengths)
        logger.info(f"查找表已保存至 {path}")

    def load(self, path: str):
        """加载查找表"""
        data = np.load(path)
        self.lut_array = data['lut_array']
        self.lut_lengths = data['lut_lengths']
        self.max_len = self.lut_array.shape[3]
        logger.info(f"查找表已加载: {path}, shape={self.lut_array.shape}")

    def get_or_build(self, cache_dir: str):
        """尝试加载或构建查找表"""
        os.makedirs(cache_dir, exist_ok=True)
        lut_path = os.path.join(cache_dir, "bresenham_lut.npz")
        if os.path.exists(lut_path):
            self.load(lut_path)
        else:
            self.build()
            self.save(lut_path)

    @staticmethod
    def _bresenham_3d(dt: int, dh: int, dw: int):
        """CPU 版 3D Bresenham，返回路径点列表（不含起止点）"""
        path = []
        steps = max(abs(dt), abs(dh), abs(dw))
        if steps == 0:
            return path

        dx = (1 if dt > 0 else -1) if dt != 0 else 0
        dy = (1 if dh > 0 else -1) if dh != 0 else 0
        dz = (1 if dw > 0 else -1) if dw != 0 else 0
        sdx, sdy, sdz = max(abs(dt), 1), max(abs(dh), 1), max(abs(dw), 1)

        x, y, z = 0, 0, 0
        ey, ez = -(steps // 2), -(steps // 2)

        for _ in range(steps):
            ey += sdy
            ez += sdz
            if ey >= 0:
                y += dy
                ey -= steps
            if ez >= 0:
                z += dz
                ez -= steps
            x += dx

            # 跳过终点
            if x == dt and y == dh and z == dw:
                break
            path.append((x, y, z))

        return path
