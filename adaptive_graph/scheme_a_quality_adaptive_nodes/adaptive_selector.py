"""
自适应节点数选择器（方案A）
============================
根据局部数据质量动态调整图中节点数量。

核心算法：
  1. 计算局部质量指标：有效像素密度 rho、空间连续性 S、时序稳定性 T
  2. 综合质量分数 Q = rho^alpha * S^beta * T^gamma
  3. 自适应节点数 N = clip(N_base * (offset + slope * Q), N_min, N_max)

所有核心计算使用 Numba @njit 加速，仅使用 numpy 数组，不使用 Python 对象。
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# 尝试导入 numba，不可用时回退到纯 Python
try:
    import numba
    from numba import njit
    HAS_NUMBA = True
    logger.info(f"[方案A] Numba 可用, 版本={numba.__version__}")
except ImportError:
    HAS_NUMBA = False
    logger.warning("[方案A] Numba 不可用, 将使用纯 Python 回退模式")


# ============================================================
# Numba JIT 内核函数
# ============================================================

if HAS_NUMBA:

    @njit(cache=True)
    def _compute_integral_image_2d(valid_slice):
        """
        计算 2D 积分图（前缀和），用于高效计算矩形区域内有效像素数。

        Parameters
        ----------
        valid_slice : (H, W) bool
            某一时间片的有效像素掩码

        Returns
        -------
        integral : (H+1, W+1) int64
            积分图，integral[i+1, j+1] = sum(valid_slice[0:i+1, 0:j+1])
        """
        H, W = valid_slice.shape
        integral = np.zeros((H + 1, W + 1), dtype=np.int64)

        for i in range(H):
            row_sum = np.int64(0)
            for j in range(W):
                if valid_slice[i, j]:
                    row_sum += np.int64(1)
                integral[i + 1, j + 1] = integral[i, j + 1] + row_sum

        return integral

    @njit(cache=True)
    def _query_integral_rect(integral, r0, c0, r1, c1):
        """
        查询积分图中矩形区域的和。

        Parameters
        ----------
        integral : (H+1, W+1) int64
        r0, c0 : int - 矩形左上角（含）
        r1, c1 : int - 矩形右下角（不含）

        Returns
        -------
        count : int64 - 矩形内有效像素数
        """
        return (integral[r1, c1] - integral[r0, c1]
                - integral[r1, c0] + integral[r0, c0])

    @njit(cache=True)
    def _compute_density_numba(valid_cube, window_r):
        """
        计算每个空间位置的有效像素密度（沿时间维度聚合后）。

        使用积分图加速：先沿时间维度求和得到 2D 有效掩码，
        再用积分图计算滑动窗口内的有效像素占比。

        Parameters
        ----------
        valid_cube : (T, H, W) bool
            时空有效掩码
        window_r : int
            滑动窗口半径

        Returns
        -------
        density : (H, W) float64
            每个空间位置的有效像素密度，范围 [0, 1]
        """
        T, H, W = valid_cube.shape

        # 沿时间维度聚合：只要任意一帧有效，该空间位置即视为有效
        spatial_valid = np.zeros((H, W), dtype=np.bool_)
        for t in range(T):
            for h in range(H):
                for w in range(W):
                    if valid_cube[t, h, w]:
                        spatial_valid[h, w] = True

        # 计算积分图
        integral = _compute_integral_image_2d(spatial_valid)

        # 滑动窗口计算密度
        density = np.zeros((H, W), dtype=np.float64)
        for h in range(H):
            for w in range(W):
                r0 = max(0, h - window_r)
                c0 = max(0, w - window_r)
                r1 = min(H, h + window_r + 1)
                c1 = min(W, w + window_r + 1)
                area = (r1 - r0) * (c1 - c0)
                if area > 0:
                    density[h, w] = float(_query_integral_rect(integral, r0, c0, r1, c1)) / float(area)
                else:
                    density[h, w] = 0.0

        return density

    @njit(cache=True)
    def _compute_spatial_continuity_numba(valid_cube, data_cube, window_r):
        """
        计算每个空间位置的空间连续性指标。

        空间连续性 S 定义为：邻域内有效像素的方差（归一化到 [0, 1]）。
        方差越小，说明有效像素分布越均匀，连续性越好。

        Parameters
        ----------
        valid_cube : (T, H, W) bool
        data_cube : (T, H, W) float32
        window_r : int

        Returns
        -------
        continuity : (H, W) float64
            空间连续性，范围 [0, 1]，值越大越连续
        """
        T, H, W = valid_cube.shape

        # 沿时间维度取均值（仅有效像素参与）
        mean_valid = np.zeros((H, W), dtype=np.float64)
        count_valid = np.zeros((H, W), dtype=np.int64)

        for t in range(T):
            for h in range(H):
                for w in range(W):
                    if valid_cube[t, h, w]:
                        mean_valid[h, w] += float(data_cube[t, h, w])
                        count_valid[h, w] += 1

        for h in range(H):
            for w in range(W):
                if count_valid[h, w] > 0:
                    mean_valid[h, w] /= float(count_valid[h, w])

        # 计算局部方差（滑动窗口内有效像素均值的方差）
        continuity = np.zeros((H, W), dtype=np.float64)
        for h in range(H):
            for w in range(W):
                r0 = max(0, h - window_r)
                c0 = max(0, w - window_r)
                r1 = min(H, h + window_r + 1)
                c1 = min(W, w + window_r + 1)

                # 收集窗口内有效均值
                vals = np.empty((r1 - r0) * (c1 - c0), dtype=np.float64)
                cnt = 0
                for hh in range(r0, r1):
                    for ww in range(c0, c1):
                        if count_valid[hh, ww] > 0:
                            vals[cnt] = mean_valid[hh, ww]
                            cnt += 1

                if cnt > 1:
                    # 计算方差
                    m = 0.0
                    for i in range(cnt):
                        m += vals[i]
                    m /= float(cnt)
                    var = 0.0
                    for i in range(cnt):
                        d = vals[i] - m
                        var += d * d
                    var /= float(cnt)

                    # 归一化：用全局数据范围作为参考
                    # 方差越小 → 连续性越好 → S 越大
                    # 使用指数衰减映射: S = exp(-k * var)
                    # k=10.0 使得 var=0.1 时 S≈0.37, var=0.01 时 S≈0.90
                    continuity[h, w] = np.exp(-10.0 * var)
                elif cnt == 1:
                    continuity[h, w] = 1.0  # 只有一个有效像素，视为完全连续
                else:
                    continuity[h, w] = 0.0  # 无有效像素

        return continuity

    @njit(cache=True)
    def _compute_temporal_stability_numba(valid_cube, data_cube, temporal_r):
        """
        计算每个空间位置的时序稳定性指标。

        时序稳定性 T 定义为：时间窗口内有效像素值的变异系数（CV）的倒数（归一化）。
        CV = std / mean，CV 越小越稳定。

        Parameters
        ----------
        valid_cube : (T, H, W) bool
        data_cube : (T, H, W) float32
        temporal_r : int
            时序窗口半径

        Returns
        -------
        stability : (H, W) float64
            时序稳定性，范围 [0, 1]，值越大越稳定
        """
        T, H, W = valid_cube.shape

        stability = np.zeros((H, W), dtype=np.float64)

        for h in range(H):
            for w in range(W):
                # 收集时间窗口内的有效值
                t0 = max(0, temporal_r)
                t1 = min(T, T - temporal_r)
                # 使用全时间范围
                t0 = 0
                t1 = T

                vals = np.empty(T, dtype=np.float64)
                cnt = 0
                for t in range(t0, t1):
                    if valid_cube[t, h, w]:
                        vals[cnt] = float(data_cube[t, h, w])
                        cnt += 1

                if cnt >= 2:
                    # 计算均值和标准差
                    m = 0.0
                    for i in range(cnt):
                        m += vals[i]
                    m /= float(cnt)

                    var = 0.0
                    for i in range(cnt):
                        d = vals[i] - m
                        var += d * d
                    var /= float(cnt)
                    std = np.sqrt(var)

                    # 变异系数 CV = std / |mean|
                    if abs(m) > 1e-8:
                        cv = std / abs(m)
                    else:
                        cv = 1.0  # 均值接近0时，认为不稳定

                    # 归一化：T = 1 / (1 + cv)，cv=0 时 T=1，cv→∞ 时 T→0
                    stability[h, w] = 1.0 / (1.0 + cv)
                elif cnt == 1:
                    stability[h, w] = 1.0  # 只有一帧，视为稳定
                else:
                    stability[h, w] = 0.0  # 无有效数据

        return stability

    @njit(cache=True)
    def _compute_quality_score_numba(density, continuity, stability,
                                      alpha, beta, gamma):
        """
        计算综合质量分数 Q = rho^alpha * S^beta * T^gamma。

        Parameters
        ----------
        density : (H, W) float64 - 有效像素密度
        continuity : (H, W) float64 - 空间连续性
        stability : (H, W) float64 - 时序稳定性
        alpha, beta, gamma : float - 权重指数

        Returns
        -------
        quality : (H, W) float64 - 综合质量分数，范围 [0, 1]
        """
        H, W = density.shape
        quality = np.zeros((H, W), dtype=np.float64)

        for h in range(H):
            for w in range(W):
                rho = density[h, w]
                s = continuity[h, w]
                t = stability[h, w]

                # 逐分量计算幂次（避免 0^0 的问题）
                q_rho = 1.0 if (rho == 0.0 and alpha == 0.0) else (rho ** alpha)
                q_s = 1.0 if (s == 0.0 and beta == 0.0) else (s ** beta)
                q_t = 1.0 if (t == 0.0 and gamma == 0.0) else (t ** gamma)

                quality[h, w] = q_rho * q_s * q_t

        return quality

    @njit(cache=True)
    def _compute_adaptive_num_nodes_numba(
        quality_map,   # (H, W) float64 - 综合质量分数
        hc, wc,        # int - 中心位置
        n_base,        # int - 基准节点数
        n_min,         # int - 最小节点数
        n_max,         # int - 最大节点数
        scale_offset,  # float - f(Q) = offset + slope * Q
        scale_slope,   # float
    ):
        """
        根据质量分数计算自适应节点数。

        N = clip(N_base * (offset + slope * Q), N_min, N_max)

        Parameters
        ----------
        quality_map : (H, W) float64
        hc, wc : int
        n_base, n_min, n_max : int
        scale_offset, scale_slope : float

        Returns
        -------
        num_nodes : int - 自适应节点数
        quality : float - 该位置的质量分数
        """
        H, W = quality_map.shape

        # 边界检查
        if hc < 0 or hc >= H or wc < 0 or wc >= W:
            return n_base, 0.0

        q = quality_map[hc, wc]

        # 限制 Q 在 [0, 1] 范围内
        if q < 0.0:
            q = 0.0
        if q > 1.0:
            q = 1.0

        # 计算自适应节点数
        f_q = scale_offset + scale_slope * q
        n_adaptive = int(round(float(n_base) * f_q))

        # 裁剪到合法范围
        if n_adaptive < n_min:
            n_adaptive = n_min
        if n_adaptive > n_max:
            n_adaptive = n_max

        # 确保至少为偶数（方便区域配额分配）
        if n_adaptive % 2 != 0:
            n_adaptive += 1
        if n_adaptive > n_max:
            n_adaptive -= 2  # 回退到偶数

        return n_adaptive, q

    @njit(cache=True)
    def _compute_all_quality_metrics_numba(
        valid_cube,    # (T, H, W) bool
        data_cube,     # (T, H, W) float32
        window_r,      # int - 空间窗口半径
        temporal_r,    # int - 时序窗口半径
        alpha, beta, gamma,  # float - 质量权重
    ):
        """
        一次性计算所有质量指标和综合质量分数。

        Parameters
        ----------
        valid_cube : (T, H, W) bool
        data_cube : (T, H, W) float32
        window_r : int
        temporal_r : int
        alpha, beta, gamma : float

        Returns
        -------
        density : (H, W) float64
        continuity : (H, W) float64
        stability : (H, W) float64
        quality : (H, W) float64
        """
        density = _compute_density_numba(valid_cube, window_r)
        continuity = _compute_spatial_continuity_numba(valid_cube, data_cube, window_r)
        stability = _compute_temporal_stability_numba(valid_cube, data_cube, temporal_r)
        quality = _compute_quality_score_numba(density, continuity, stability,
                                                alpha, beta, gamma)
        return density, continuity, stability, quality


# ============================================================
# 纯 Python 回退（Numba 不可用时）
# ============================================================

else:

    def _compute_density_python(valid_cube, window_r):
        """纯 Python 回退：计算有效像素密度"""
        T, H, W = valid_cube.shape
        spatial_valid = np.any(valid_cube, axis=0)  # (H, W)

        # 使用 scipy 或手动积分图
        from scipy.ndimage import uniform_filter
        density = uniform_filter(spatial_valid.astype(np.float64), size=2*window_r+1, mode='constant')
        return density

    def _compute_spatial_continuity_python(valid_cube, data_cube, window_r):
        """纯 Python 回退：计算空间连续性"""
        T, H, W = valid_cube.shape

        # 沿时间维度取均值
        valid_count = np.sum(valid_cube, axis=0).astype(np.float64)
        data_sum = np.nansum(np.where(valid_cube, data_cube, 0.0), axis=0)
        mean_valid = np.divide(data_sum, valid_count,
                               out=np.zeros((H, W)), where=valid_count > 0)

        # 局部方差
        from scipy.ndimage import uniform_filter
        mean_sq = uniform_filter(mean_valid ** 2, size=2*window_r+1, mode='constant')
        mean_mean = uniform_filter(mean_valid, size=2*window_r+1, mode='constant')
        var = np.maximum(mean_sq - mean_mean ** 2, 0.0)

        continuity = np.exp(-10.0 * var)
        continuity[valid_count == 0] = 0.0
        continuity[(valid_count > 0) & (valid_count <= 1)] = 1.0
        return continuity

    def _compute_temporal_stability_python(valid_cube, data_cube, temporal_r):
        """纯 Python 回退：计算时序稳定性"""
        T, H, W = valid_cube.shape
        stability = np.zeros((H, W), dtype=np.float64)

        for h in range(H):
            for w in range(W):
                vals = data_cube[:, h, w][valid_cube[:, h, w]]
                if len(vals) >= 2:
                    m = np.mean(vals)
                    std = np.std(vals)
                    cv = std / max(abs(m), 1e-8)
                    stability[h, w] = 1.0 / (1.0 + cv)
                elif len(vals) == 1:
                    stability[h, w] = 1.0

        return stability

    def _compute_quality_score_python(density, continuity, stability,
                                       alpha, beta, gamma):
        """纯 Python 回退：计算综合质量分数"""
        with np.errstate(divide='ignore', invalid='ignore'):
            q_rho = np.power(density, alpha, where=density > 0)
            q_rho = np.where(density == 0, 0.0 if alpha > 0 else 1.0, q_rho)
            q_s = np.power(continuity, beta, where=continuity > 0)
            q_s = np.where(continuity == 0, 0.0 if beta > 0 else 1.0, q_s)
            q_t = np.power(stability, gamma, where=stability > 0)
            q_t = np.where(stability == 0, 0.0 if gamma > 0 else 1.0, q_t)
        return np.clip(q_rho * q_s * q_t, 0.0, 1.0)

    def _compute_adaptive_num_nodes_python(
        quality_map, hc, wc, n_base, n_min, n_max,
        scale_offset, scale_slope,
    ):
        """纯 Python 回退：计算自适应节点数"""
        H, W = quality_map.shape
        if hc < 0 or hc >= H or wc < 0 or wc >= W:
            return n_base, 0.0

        q = float(np.clip(quality_map[hc, wc], 0.0, 1.0))
        f_q = scale_offset + scale_slope * q
        n_adaptive = int(round(n_base * f_q))
        n_adaptive = max(n_min, min(n_max, n_adaptive))

        # 确保偶数
        if n_adaptive % 2 != 0:
            n_adaptive += 1
            if n_adaptive > n_max:
                n_adaptive -= 2

        return n_adaptive, q

    def _compute_all_quality_metrics_python(
        valid_cube, data_cube, window_r, temporal_r,
        alpha, beta, gamma,
    ):
        """纯 Python 回退：一次性计算所有质量指标"""
        density = _compute_density_python(valid_cube, window_r)
        continuity = _compute_spatial_continuity_python(valid_cube, data_cube, window_r)
        stability = _compute_temporal_stability_python(valid_cube, data_cube, temporal_r)
        quality = _compute_quality_score_python(density, continuity, stability,
                                                 alpha, beta, gamma)
        return density, continuity, stability, quality


# ============================================================
# 高层 Python 接口（包装 Numba/Python 回退）
# ============================================================

class AdaptiveNodeSelector:
    """
    自适应节点数选择器。

    根据局部数据质量动态决定每个位置的最优节点数量。

    使用方法：
        selector = AdaptiveNodeSelector(quality_config)
        selector.precompute(valid_mask, data)  # 预计算全局质量图
        n, q = selector.get_num_nodes(hc, wc)  # 查询某位置的节点数
    """

    def __init__(self, quality_config):
        """
        Parameters
        ----------
        quality_config : QualityAdaptiveConfig
            质量自适应参数配置
        """
        self.cfg = quality_config
        self._quality_map = None       # (H, W) float64 - 综合质量分数
        self._density_map = None       # (H, W) float64 - 有效像素密度
        self._continuity_map = None    # (H, W) float64 - 空间连续性
        self._stability_map = None     # (H, W) float64 - 时序稳定性
        self._precomputed = False

    def precompute(self, valid_mask, data):
        """
        预计算全局质量图。

        在构建所有图之前调用一次，后续通过 get_num_nodes 快速查询。

        Parameters
        ----------
        valid_mask : (T, H, W) bool
            全局有效像素掩码
        data : (T, H, W) float32
            全局数据
        """
        logger.info("[方案A] 开始预计算质量指标图...")

        if HAS_NUMBA:
            self._density_map, self._continuity_map, self._stability_map, self._quality_map = \
                _compute_all_quality_metrics_numba(
                    valid_mask.astype(np.bool_),
                    data.astype(np.float32),
                    self.cfg.integral_window,
                    self.cfg.temporal_window,
                    self.cfg.alpha,
                    self.cfg.beta,
                    self.cfg.gamma,
                )
        else:
            self._density_map, self._continuity_map, self._stability_map, self._quality_map = \
                _compute_all_quality_metrics_python(
                    valid_mask, data,
                    self.cfg.integral_window,
                    self.cfg.temporal_window,
                    self.cfg.alpha,
                    self.cfg.beta,
                    self.cfg.gamma,
                )

        self._precomputed = True

        # 统计信息
        q = self._quality_map
        logger.info(f"[方案A] 质量指标预计算完成:")
        logger.info(f"  密度      - 均值: {self._density_map.mean():.3f}, "
                     f"范围: [{self._density_map.min():.3f}, {self._density_map.max():.3f}]")
        logger.info(f"  连续性    - 均值: {self._continuity_map.mean():.3f}, "
                     f"范围: [{self._continuity_map.min():.3f}, {self._continuity_map.max():.3f}]")
        logger.info(f"  稳定性    - 均值: {self._stability_map.mean():.3f}, "
                     f"范围: [{self._stability_map.min():.3f}, {self._stability_map.max():.3f}]")
        logger.info(f"  综合质量Q - 均值: {q.mean():.3f}, "
                     f"范围: [{q.min():.3f}, {q.max():.3f}]")

    def precompute_from_cube(self, valid_cube, data_cube, tc, hc, wc):
        """
        从局部子立方体预计算质量（用于单次构建场景）。

        Parameters
        ----------
        valid_cube : (ct, ch, cw) bool
        data_cube : (ct, ch, cw) float32
        tc, hc, wc : int
            中心在全局坐标中的位置
        """
        # 对于局部子立方体，直接计算中心位置的质量
        ch_local = valid_cube.shape[1] // 2
        cw_local = valid_cube.shape[2] // 2

        if HAS_NUMBA:
            density, continuity, stability, quality = \
                _compute_all_quality_metrics_numba(
                    valid_cube.astype(np.bool_),
                    data_cube.astype(np.float32),
                    self.cfg.integral_window,
                    self.cfg.temporal_window,
                    self.cfg.alpha,
                    self.cfg.beta,
                    self.cfg.gamma,
                )
        else:
            density, continuity, stability, quality = \
                _compute_all_quality_metrics_python(
                    valid_cube, data_cube,
                    self.cfg.integral_window,
                    self.cfg.temporal_window,
                    self.cfg.alpha,
                    self.cfg.beta,
                    self.cfg.gamma,
                )

        self._density_map = density
        self._continuity_map = continuity
        self._stability_map = stability
        self._quality_map = quality
        self._precomputed = True

        return quality[ch_local, cw_local]

    def get_num_nodes(self, hc, wc, n_base):
        """
        查询指定位置的自适应节点数。

        Parameters
        ----------
        hc, wc : int
            空间位置坐标
        n_base : int
            基准节点数

        Returns
        -------
        num_nodes : int
            自适应节点数
        quality : float
            该位置的质量分数
        """
        if not self._precomputed:
            logger.warning("[方案A] 质量图未预计算，使用基准节点数")
            return n_base, 0.0

        if HAS_NUMBA:
            n, q = _compute_adaptive_num_nodes_numba(
                self._quality_map, hc, wc,
                n_base,
                self.cfg.n_min,
                self.cfg.n_max,
                self.cfg.scale_offset,
                self.cfg.scale_slope,
            )
        else:
            n, q = _compute_adaptive_num_nodes_python(
                self._quality_map, hc, wc,
                n_base,
                self.cfg.n_min,
                self.cfg.n_max,
                self.cfg.scale_offset,
                self.cfg.scale_slope,
            )

        return n, q

    def get_quality_map(self):
        """返回综合质量分数图 (H, W)"""
        return self._quality_map

    def get_density_map(self):
        """返回有效像素密度图 (H, W)"""
        return self._density_map

    def get_continuity_map(self):
        """返回空间连续性图 (H, W)"""
        return self._continuity_map

    def get_stability_map(self):
        """返回时序稳定性图 (H, W)"""
        return self._stability_map

    @staticmethod
    def warmup():
        """预热 Numba JIT 编译（在初始化时调用）"""
        if not HAS_NUMBA:
            return

        tiny_valid = np.ones((5, 5, 5), dtype=np.bool_)
        tiny_data = np.random.rand(5, 5, 5).astype(np.float32)

        # 触发所有 JIT 函数的编译
        _compute_density_numba(tiny_valid, 1)
        _compute_spatial_continuity_numba(tiny_valid, tiny_data, 1)
        _compute_temporal_stability_numba(tiny_valid, tiny_data, 1)
        _compute_quality_score_numba(
            np.ones((5, 5), dtype=np.float64),
            np.ones((5, 5), dtype=np.float64),
            np.ones((5, 5), dtype=np.float64),
            0.5, 0.3, 0.2,
        )
        _compute_adaptive_num_nodes_numba(
            np.ones((5, 5), dtype=np.float64),
            2, 2, 36, 18, 54, 0.5, 1.0,
        )
        _compute_all_quality_metrics_numba(
            tiny_valid, tiny_data, 1, 1, 0.5, 0.3, 0.2,
        )
