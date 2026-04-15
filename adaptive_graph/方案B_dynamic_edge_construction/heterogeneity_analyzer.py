"""
空间异质性分析器模块
====================
计算局部时空立方体内的空间异质性指数 H = std(NTL_values) / mean(NTL_values)。

核心功能：
  1. 使用 Numba @njit 加速异质性指数的逐位置计算
  2. 支持批量计算（对多个中心位置同时计算）
  3. 根据异质性指数计算动态时空权重 w_spatial 和 w_temporal

算法逻辑：
  - H = std(NTL_values) / mean(NTL_values)  （变异系数）
  - w_spatial = sigmoid((H - H_threshold) * scale)
  - w_temporal = 1 - w_spatial
  - H 高 → w_spatial 大 → 城市核心区域，侧重空间连接
  - H 低 → w_temporal 大 → 郊区区域，侧重时序连接
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# 尝试导入 numba，不可用时回退到纯 Python
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
    logger.info(f"[方案B] Numba 可用, 版本={numba.__version__}")
except ImportError:
    HAS_NUMBA = False
    logger.warning("[方案B] Numba 不可用, 异质性分析器将使用纯 Python 回退模式")


# ============================================================
# Numba JIT 内核函数
# ============================================================

if HAS_NUMBA:

    @njit(cache=True)
    def _sigmoid(x):
        """数值稳定的 sigmoid 函数"""
        # 对大正数和大负数做截断，避免溢出
        if x >= 20.0:
            return 1.0
        elif x <= -20.0:
            return 0.0
        else:
            return 1.0 / (1.0 + np.exp(-x))

    @njit(cache=True)
    def _compute_heterogeneity_single(
        data_cube,      # (ct_size, ch_size, cw_size) float32
        valid_cube,     # (ct_size, ch_size, cw_size) bool
        min_valid,      # int - 计算异质性所需的最小有效像素数
    ):
        """
        计算单个时空立方体内的空间异质性指数。

        算法：
          H = std(NTL_values) / mean(NTL_values)

        仅在空间维度（H, W）上计算变异系数，时间维度取均值以获得
        更稳定的空间异质性表征。

        Parameters
        ----------
        data_cube : (ct_size, ch_size, cw_size) float32
            时空立方体内的 NTL 数据
        valid_cube : (ct_size, ch_size, cw_size) bool
            有效像素掩码
        min_valid : int
            最小有效像素数，低于此值返回 -1（无效）

        Returns
        -------
        H : float
            空间异质性指数。返回 -1.0 表示数据不足，无法计算。
        """
        ct_size, ch_size, cw_size = data_cube.shape

        # 第一遍：计算每个空间位置的时间均值
        # spatial_mean(h, w) = mean_t(data_cube[t, h, w])
        spatial_values = np.empty(ch_size * cw_size, dtype=np.float32)
        spatial_count = 0

        for h in range(ch_size):
            for w in range(cw_size):
                # 计算该空间位置在所有时间步上的均值
                t_sum = 0.0
                t_count = 0
                for t in range(ct_size):
                    if valid_cube[t, h, w]:
                        t_sum += data_cube[t, h, w]
                        t_count += 1

                if t_count > 0:
                    spatial_values[spatial_count] = t_sum / t_count
                    spatial_count += 1

        # 有效空间位置不足
        if spatial_count < min_valid:
            return -1.0

        # 第二遍：计算空间均值
        mean_val = 0.0
        for i in range(spatial_count):
            mean_val += spatial_values[i]
        mean_val /= spatial_count

        # 均值接近零时无法计算有意义的变异系数
        if mean_val < 1e-6:
            return -1.0

        # 第三遍：计算空间标准差
        var_val = 0.0
        for i in range(spatial_count):
            diff = spatial_values[i] - mean_val
            var_val += diff * diff
        var_val /= spatial_count
        std_val = np.sqrt(var_val)

        # 变异系数 H = std / mean
        H = std_val / mean_val

        return H

    @njit(cache=True)
    def _compute_spatial_temporal_weights(
        H,              # float - 异质性指数
        h_threshold,    # float - 异质性阈值
        h_scale,        # float - sigmoid 缩放因子
    ):
        """
        根据异质性指数计算动态时空权重。

        Parameters
        ----------
        H : float
            空间异质性指数（-1.0 表示无效）
        h_threshold : float
            异质性阈值
        h_scale : float
            sigmoid 缩放因子

        Returns
        -------
        w_spatial : float
            空间维度权重 [0, 1]
        w_temporal : float
            时序维度权重 [0, 1]，等于 1 - w_spatial
        """
        if H < 0.0:
            # 异质性无效时，使用默认均衡权重
            return 0.5, 0.5

        w_spatial = _sigmoid((H - h_threshold) * h_scale)
        w_temporal = 1.0 - w_spatial

        return w_spatial, w_temporal

    @njit(cache=True)
    def _compute_heterogeneity_batch(
        full_data,          # (T, H, W) float32 - 完整 NTL 数据
        full_valid,         # (T, H, W) bool - 完整有效掩码
        center_positions,   # (num_centers, 3) int32 - 中心位置 (t, h, w)
        cube_radius,        # int - 局部立方体半径
        min_valid,          # int - 最小有效像素数
        h_threshold,        # float - 异质性阈值
        h_scale,            # float - sigmoid 缩放因子
    ):
        """
        批量计算多个中心位置的异质性指数和时空权重。

        Parameters
        ----------
        full_data : (T, H, W) float32
        full_valid : (T, H, W) bool
        center_positions : (num_centers, 3) int32
        cube_radius : int
        min_valid : int
        h_threshold : float
        h_scale : float

        Returns
        -------
        heterogeneity_indices : (num_centers,) float32
            每个中心位置的异质性指数
        w_spatial_arr : (num_centers,) float32
            每个中心位置的空间权重
        w_temporal_arr : (num_centers,) float32
            每个中心位置的时序权重
        """
        T = full_data.shape[0]
        H_dim = full_data.shape[1]
        W_dim = full_data.shape[2]
        num_centers = center_positions.shape[0]

        heterogeneity_indices = np.empty(num_centers, dtype=np.float32)
        w_spatial_arr = np.empty(num_centers, dtype=np.float32)
        w_temporal_arr = np.empty(num_centers, dtype=np.float32)

        for c in range(num_centers):
            tc = center_positions[c, 0]
            hc = center_positions[c, 1]
            wc = center_positions[c, 2]

            # 裁剪局部立方体边界
            t0 = max(0, tc - cube_radius)
            t1 = min(T, tc + cube_radius + 1)
            h0 = max(0, hc - cube_radius)
            h1 = min(H_dim, hc + cube_radius + 1)
            w0 = max(0, wc - cube_radius)
            w1 = min(W_dim, wc + cube_radius + 1)

            # 提取子立方体
            sub_data = full_data[t0:t1, h0:h1, w0:w1]
            sub_valid = full_valid[t0:t1, h0:h1, w0:w1]

            # 计算异质性指数
            H_val = _compute_heterogeneity_single(sub_data, sub_valid, min_valid)
            heterogeneity_indices[c] = H_val

            # 计算时空权重
            ws, wt = _compute_spatial_temporal_weights(H_val, h_threshold, h_scale)
            w_spatial_arr[c] = ws
            w_temporal_arr[c] = wt

        return heterogeneity_indices, w_spatial_arr, w_temporal_arr

    @njit(cache=True)
    def _compute_heterogeneity_for_cube(
        data_cube,      # (ct_size, ch_size, cw_size) float32
        valid_cube,     # (ct_size, ch_size, cw_size) bool
        min_valid,      # int
        h_threshold,    # float
        h_scale,        # float
    ):
        """
        为单个已裁剪的立方体计算异质性指数和时空权重。
        供 GraphBuilderB 在 build_single 中直接调用。

        Returns
        -------
        H : float
        w_spatial : float
        w_temporal : float
        """
        H_val = _compute_heterogeneity_single(data_cube, valid_cube, min_valid)
        ws, wt = _compute_spatial_temporal_weights(H_val, h_threshold, h_scale)
        return H_val, ws, wt


# ============================================================
# 纯 Python 回退（Numba 不可用时）
# ============================================================

else:

    def _sigmoid(x):
        """数值稳定的 sigmoid 函数（纯 Python）"""
        x = np.clip(x, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _compute_heterogeneity_single(data_cube, valid_cube, min_valid):
        """纯 Python 回退：计算单个立方体的异质性指数"""
        ct_size, ch_size, cw_size = data_cube.shape

        # 计算每个空间位置的时间均值
        spatial_values = []
        for h in range(ch_size):
            for w in range(cw_size):
                valid_vals = data_cube[:, h, w][valid_cube[:, h, w]]
                if len(valid_vals) > 0:
                    spatial_values.append(np.mean(valid_vals))

        if len(spatial_values) < min_valid:
            return -1.0

        mean_val = np.mean(spatial_values)
        if mean_val < 1e-6:
            return -1.0

        std_val = np.std(spatial_values)
        return std_val / mean_val

    def _compute_spatial_temporal_weights(H, h_threshold, h_scale):
        """纯 Python 回退：计算时空权重"""
        if H < 0.0:
            return 0.5, 0.5
        w_spatial = float(_sigmoid((H - h_threshold) * h_scale))
        w_temporal = 1.0 - w_spatial
        return w_spatial, w_temporal

    def _compute_heterogeneity_batch(
        full_data, full_valid, center_positions,
        cube_radius, min_valid, h_threshold, h_scale
    ):
        """纯 Python 回退：批量计算异质性"""
        T, H_dim, W_dim = full_data.shape
        num_centers = len(center_positions)
        heterogeneity_indices = np.zeros(num_centers, dtype=np.float32)
        w_spatial_arr = np.zeros(num_centers, dtype=np.float32)
        w_temporal_arr = np.zeros(num_centers, dtype=np.float32)

        for c in range(num_centers):
            tc, hc, wc = center_positions[c]
            t0 = max(0, tc - cube_radius)
            t1 = min(T, tc + cube_radius + 1)
            h0 = max(0, hc - cube_radius)
            h1 = min(H_dim, hc + cube_radius + 1)
            w0 = max(0, wc - cube_radius)
            w1 = min(W_dim, wc + cube_radius + 1)

            sub_data = full_data[t0:t1, h0:h1, w0:w1]
            sub_valid = full_valid[t0:t1, h0:h1, w0:w1]

            H_val = _compute_heterogeneity_single(sub_data, sub_valid, min_valid)
            heterogeneity_indices[c] = H_val
            ws, wt = _compute_spatial_temporal_weights(H_val, h_threshold, h_scale)
            w_spatial_arr[c] = ws
            w_temporal_arr[c] = wt

        return heterogeneity_indices, w_spatial_arr, w_temporal_arr

    def _compute_heterogeneity_for_cube(
        data_cube, valid_cube, min_valid, h_threshold, h_scale
    ):
        """纯 Python 回退：单立方体异质性+权重"""
        H_val = _compute_heterogeneity_single(data_cube, valid_cube, min_valid)
        ws, wt = _compute_spatial_temporal_weights(H_val, h_threshold, h_scale)
        return H_val, ws, wt


# ============================================================
# 高级接口类
# ============================================================

class HeterogeneityAnalyzer:
    """
    空间异质性分析器（高级接口）。

    封装 Numba JIT 内核，提供便捷的 Python 接口。
    支持单位置计算和批量计算两种模式。

    使用示例：
        >>> analyzer = HeterogeneityAnalyzer(
        ...     cube_radius=5, h_threshold=0.25, h_scale=10.0, min_valid=8
        ... )
        >>> H, ws, wt = analyzer.analyze_cube(data_cube, valid_cube)
        >>> print(f"异质性指数: {H:.3f}, 空间权重: {ws:.3f}, 时序权重: {wt:.3f}")
    """

    def __init__(
        self,
        cube_radius: int = 5,
        h_threshold: float = 0.25,
        h_scale: float = 10.0,
        min_valid: int = 8,
    ):
        """
        Parameters
        ----------
        cube_radius : int
            计算异质性时使用的局部立方体半径
        h_threshold : float
            异质性阈值，用于 sigmoid 权重计算
        h_scale : float
            sigmoid 缩放因子
        min_valid : int
            最小有效像素数
        """
        self.cube_radius = cube_radius
        self.h_threshold = h_threshold
        self.h_scale = h_scale
        self.min_valid = min_valid

    def analyze_cube(self, data_cube: np.ndarray, valid_cube: np.ndarray):
        """
        分析单个已裁剪的时空立方体。

        Parameters
        ----------
        data_cube : (ct_size, ch_size, cw_size) float32
        valid_cube : (ct_size, ch_size, cw_size) bool

        Returns
        -------
        H : float
            异质性指数（-1.0 表示无效）
        w_spatial : float
            空间维度权重
        w_temporal : float
            时序维度权重
        """
        return _compute_heterogeneity_for_cube(
            data_cube, valid_cube,
            self.min_valid, self.h_threshold, self.h_scale
        )

    def analyze_batch(
        self,
        full_data: np.ndarray,
        full_valid: np.ndarray,
        center_positions: np.ndarray,
    ):
        """
        批量分析多个中心位置的异质性。

        Parameters
        ----------
        full_data : (T, H, W) float32
            完整 NTL 数据
        full_valid : (T, H, W) bool
            完整有效掩码
        center_positions : (N, 3) int32
            中心位置数组

        Returns
        -------
        heterogeneity_indices : (N,) float32
        w_spatial_arr : (N,) float32
        w_temporal_arr : (N,) float32
        """
        return _compute_heterogeneity_batch(
            full_data, full_valid,
            center_positions.astype(np.int32),
            self.cube_radius, self.min_valid,
            self.h_threshold, self.h_scale
        )

    def is_high_heterogeneity(self, H: float) -> bool:
        """判断是否为高异质性区域"""
        return H >= self.h_threshold

    def is_low_heterogeneity(self, H: float) -> bool:
        """判断是否为低异质性区域"""
        return 0.0 <= H < self.h_threshold
