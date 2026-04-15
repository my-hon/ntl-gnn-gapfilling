"""
动态边构建器模块
================
基于空间异质性指数，动态调整 Bresenham 路径过滤策略和边属性。

核心改进（相比 v2 的 _build_edges_numba）：
  1. 动态权重边属性：
     edge_attr = w_spatial * spatial_offset + w_temporal * temporal_offset
     高异质性区域：空间偏移权重大，保留空间结构信息
     低异质性区域：时序偏移权重大，保留时间演变信息

  2. 高异质性区域 - 放松 Bresenham 截断条件：
     增加 spatial_boost 系数，允许更长的空间路径通过，
     使高异质性的城市核心区域能建立更多空间近邻连接。

  3. 低异质性区域 - 扩展时序连接窗口：
     增加 temporal_extend 步数，允许连接更远时间步的节点，
     使低异质性的郊区区域能捕获更长的时间依赖关系。

灵感来源：
  - Graph WaveNet (arXiv:1906.00121): 自适应邻接矩阵学习
  - DRTR (arXiv:2406.17281): 距离感知拓扑精炼
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# 尝试导入 numba
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logger.warning("[方案B] Numba 不可用, 动态边构建器将使用纯 Python 回退模式")


# ============================================================
# Numba JIT 内核函数
# ============================================================

if HAS_NUMBA:

    @njit(cache=True)
    def _build_edges_dynamic_numba(
        node_offsets,        # (N, 3) int32 - 节点偏移量
        valid_cube,          # (ct_size, ch_size, cw_size) bool - 有效掩码
        lut_array,           # (2R+1, 2R+1, 2R+1, max_len, 3) int16 - Bresenham查找表
        lut_lengths,         # (2R+1, 2R+1, 2R+1) int16 - 路径长度
        lut_radius,          # int - 查找表半径
        w_spatial,           # float - 空间维度权重
        w_temporal,          # float - 时序维度权重
        spatial_boost,       # float - 高异质性区域的路径放松系数
        temporal_extend,     # int - 低异质性区域的时序扩展步数
    ):
        """
        方案B核心：基于异质性权重的动态边构建（Numba JIT 加速）。

        相比 v2 的 _build_edges_numba，主要改动：
        1. 边属性使用动态权重：edge_attr = w_spatial * spatial_offset + w_temporal * temporal_offset
        2. 高异质性时放松 Bresenham 截断（通过 spatial_boost）
        3. 低异质性时扩展时序连接（通过 temporal_extend）

        Parameters
        ----------
        node_offsets : (N, 3) int32
        valid_cube : (ct_size, ch_size, cw_size) bool
        lut_array : Bresenham 查找表
        lut_lengths : 路径长度表
        lut_radius : int
        w_spatial : float - 空间权重 [0, 1]
        w_temporal : float - 时序权重 [0, 1]
        spatial_boost : float - 路径放松系数（>1.0 放松截断）
        temporal_extend : int - 时序扩展步数（>=0）

        Returns
        -------
        edge_src : (E,) int64
        edge_dst : (E,) int64
        edge_attrs : (E, 3) float32
        num_edges : int
        """
        N = node_offsets.shape[0]
        R = lut_radius
        ct = valid_cube.shape[0] // 2
        ch = valid_cube.shape[1] // 2
        cw = valid_cube.shape[2] // 2
        ct_size = valid_cube.shape[0]
        ch_size = valid_cube.shape[1]
        cw_size = valid_cube.shape[2]

        # 构建偏移量 → 节点索引的查找表
        lut_size = 2 * R + 1
        offset_lut = np.full((lut_size, lut_size, lut_size), -1, dtype=np.int32)
        for i in range(N):
            dt = node_offsets[i, 0]
            dh = node_offsets[i, 1]
            dw = node_offsets[i, 2]
            if (0 <= dt + R < lut_size and
                0 <= dh + R < lut_size and
                0 <= dw + R < lut_size):
                offset_lut[dt + R, dh + R, dw + R] = i

        # 预分配边数组
        # 最大边数估算：每个非中心节点最多 2 条边（Bresenham + 时序扩展）
        # 加上自环和额外空间连接
        max_edges = 4 * N
        edge_src = np.zeros(max_edges, dtype=np.int64)
        edge_dst = np.zeros(max_edges, dtype=np.int64)
        edge_attrs = np.zeros((max_edges, 3), dtype=np.float32)
        num_edges = 0

        for i in range(N):
            dt = node_offsets[i, 0]
            dh = node_offsets[i, 1]
            dw = node_offsets[i, 2]

            # 跳过中心节点
            if dt == 0 and dh == 0 and dw == 0:
                continue

            li = dt + R
            lj = dh + R
            lk = dw + R
            path_len = lut_lengths[li, lj, lk]

            first_valid = -1
            spatial_dist = 0  # 空间距离（用于判断是否为空间近邻）

            if path_len > 0:
                # 计算有效路径的允许长度（高异质性时放松截断）
                effective_path_len = int(path_len * spatial_boost)

                for m in range(min(path_len, effective_path_len)):
                    pt = int(lut_array[li, lj, lk, m, 0])
                    ph = int(lut_array[li, lj, lk, m, 1])
                    pw = int(lut_array[li, lj, lk, m, 2])

                    # 检查是否在有效范围内
                    gt = pt + ct
                    gh = ph + ch
                    gw = pw + cw
                    if gt < 0 or gt >= ct_size or gh < 0 or gh >= ch_size or gw < 0 or gw >= cw_size:
                        break

                    if not valid_cube[gt, gh, gw]:
                        break

                    # 检查该位置是否是已选节点
                    if (0 <= pt + R < lut_size and
                        0 <= ph + R < lut_size and
                        0 <= pw + R < lut_size):
                        node_idx = offset_lut[pt + R, ph + R, pw + R]
                        if node_idx >= 0:
                            first_valid = node_idx
                            spatial_dist = m + 1  # Bresenham 路径步数作为空间距离
                            break

            # ---- 添加主边（Bresenham 路径边）----
            if first_valid >= 0:
                # 计算空间偏移和时序偏移
                src_spatial = np.float32(dh - node_offsets[first_valid, 1])
                src_spatial_w = np.float32(dw - node_offsets[first_valid, 2])
                src_temporal = np.float32(dt - node_offsets[first_valid, 0])

                # 动态权重边属性
                # 空间偏移 = (dh差, dw差)，时序偏移 = dt差
                edge_attrs[num_edges, 0] = w_spatial * src_temporal + w_temporal * src_temporal
                edge_attrs[num_edges, 1] = w_spatial * src_spatial + w_temporal * 0.0
                edge_attrs[num_edges, 2] = w_spatial * src_spatial_w + w_temporal * 0.0

                edge_src[num_edges] = i
                edge_dst[num_edges] = first_valid
                num_edges += 1
            else:
                # 无遮挡，直接连向中心节点
                edge_attrs[num_edges, 0] = w_spatial * np.float32(dt) + w_temporal * np.float32(dt)
                edge_attrs[num_edges, 1] = w_spatial * np.float32(dh)
                edge_attrs[num_edges, 2] = w_spatial * np.float32(dw)

                edge_src[num_edges] = i
                edge_dst[num_edges] = 0
                num_edges += 1

            # ---- 低异质性区域：扩展时序连接 ----
            # 当 w_temporal > 0.5 时，尝试连接相同空间位置的相邻时间步节点
            if w_temporal > 0.5 and temporal_extend > 0:
                for t_ext in range(1, temporal_extend + 1):
                    if num_edges >= max_edges:
                        break

                    # 正向时间扩展
                    ext_dt = dt + t_ext
                    if (0 <= ext_dt + R < lut_size and
                        0 <= dh + R < lut_size and
                        0 <= dw + R < lut_size):
                        ext_idx = offset_lut[ext_dt + R, dh + R, dw + R]
                        if ext_idx >= 0:
                            # 时序扩展边：属性主要反映时间偏移
                            edge_attrs[num_edges, 0] = w_temporal * np.float32(t_ext)
                            edge_attrs[num_edges, 1] = 0.0
                            edge_attrs[num_edges, 2] = 0.0
                            edge_src[num_edges] = i
                            edge_dst[num_edges] = ext_idx
                            num_edges += 1

                    # 反向时间扩展
                    ext_dt = dt - t_ext
                    if (0 <= ext_dt + R < lut_size and
                        0 <= dh + R < lut_size and
                        0 <= dw + R < lut_size):
                        ext_idx = offset_lut[ext_dt + R, dh + R, dw + R]
                        if ext_idx >= 0:
                            edge_attrs[num_edges, 0] = w_temporal * np.float32(-t_ext)
                            edge_attrs[num_edges, 1] = 0.0
                            edge_attrs[num_edges, 2] = 0.0
                            edge_src[num_edges] = i
                            edge_dst[num_edges] = ext_idx
                            num_edges += 1

            # ---- 高异质性区域：增加空间近邻连接 ----
            # 当 w_spatial > 0.5 时，对空间近邻（曼哈顿距离 <= 2）增加额外边
            if w_spatial > 0.5 and spatial_boost > 1.0:
                # 检查 6 个直接空间邻居
                spatial_neighbors = np.array([
                    [dt, dh + 1, dw],
                    [dt, dh - 1, dw],
                    [dt, dh, dw + 1],
                    [dt, dh, dw - 1],
                    [dt + 1, dh, dw],  # 也包含时间邻居
                    [dt - 1, dh, dw],
                ], dtype=np.int32)

                for sn in range(spatial_neighbors.shape[0]):
                    if num_edges >= max_edges:
                        break

                    sndt = spatial_neighbors[sn, 0]
                    sndh = spatial_neighbors[sn, 1]
                    sndw = spatial_neighbors[sn, 2]

                    if (0 <= sndt + R < lut_size and
                        0 <= sndh + R < lut_size and
                        0 <= sndw + R < lut_size):
                        sn_idx = offset_lut[sndt + R, sndh + R, sndw + R]
                        if sn_idx >= 0 and sn_idx != i:
                            # 检查是否已经存在这条边（避免重复）
                            dup = False
                            for e in range(num_edges):
                                if (edge_src[e] == i and edge_dst[e] == sn_idx):
                                    dup = True
                                    break
                            if not dup:
                                # 空间近邻边：属性主要反映空间偏移
                                edge_attrs[num_edges, 0] = w_spatial * np.float32(sndt - dt)
                                edge_attrs[num_edges, 1] = w_spatial * np.float32(sndh - dh)
                                edge_attrs[num_edges, 2] = w_spatial * np.float32(sndw - dw)
                                edge_src[num_edges] = i
                                edge_dst[num_edges] = sn_idx
                                num_edges += 1

        return (edge_src[:num_edges],
                edge_dst[:num_edges],
                edge_attrs[:num_edges],
                num_edges)

    @njit(cache=True)
    def _build_edges_dynamic_with_heterogeneity(
        node_offsets,        # (N, 3) int32
        valid_cube,          # (ct_size, ch_size, cw_size) bool
        data_cube,           # (ct_size, ch_size, cw_size) float32
        lut_array,           # Bresenham 查找表
        lut_lengths,         # 路径长度表
        lut_radius,          # int
        h_threshold,         # float - 异质性阈值
        h_scale,             # float - sigmoid 缩放因子
        spatial_boost,       # float - 路径放松系数
        temporal_extend,     # int - 时序扩展步数
        min_valid,           # int - 最小有效像素数
    ):
        """
        完整版动态边构建：内部自动计算异质性指数和权重。

        将异质性分析和边构建合并为一个 Numba JIT 函数，
        避免额外的 Python-Numba 数据传输开销。

        Parameters
        ----------
        node_offsets : (N, 3) int32
        valid_cube : (ct_size, ch_size, cw_size) bool
        data_cube : (ct_size, ch_size, cw_size) float32
        lut_array, lut_lengths, lut_radius : Bresenham 查找表
        h_threshold, h_scale : 异质性参数
        spatial_boost, temporal_extend : 动态调整参数
        min_valid : int

        Returns
        -------
        edge_src, edge_dst, edge_attrs, num_edges : 同 _build_edges_dynamic_numba
        H_value : float - 计算得到的异质性指数
        w_spatial : float - 空间权重
        w_temporal : float - 时序权重
        """
        ct_size, ch_size, cw_size = data_cube.shape

        # ---- 内联计算异质性指数 ----
        # 计算每个空间位置的时间均值
        spatial_values = np.empty(ch_size * cw_size, dtype=np.float32)
        spatial_count = 0

        for h in range(ch_size):
            for w in range(cw_size):
                t_sum = 0.0
                t_cnt = 0
                for t in range(ct_size):
                    if valid_cube[t, h, w]:
                        t_sum += data_cube[t, h, w]
                        t_cnt += 1
                if t_cnt > 0:
                    spatial_values[spatial_count] = t_sum / t_cnt
                    spatial_count += 1

        # 计算变异系数 H
        H_value = -1.0
        if spatial_count >= min_valid:
            mean_val = 0.0
            for i in range(spatial_count):
                mean_val += spatial_values[i]
            mean_val /= spatial_count

            if mean_val >= 1e-6:
                var_val = 0.0
                for i in range(spatial_count):
                    diff = spatial_values[i] - mean_val
                    var_val += diff * diff
                var_val /= spatial_count
                H_value = np.sqrt(var_val) / mean_val

        # 计算 sigmoid 权重
        if H_value < 0.0:
            w_spatial = 0.5
            w_temporal = 0.5
        else:
            sig_arg = (H_value - h_threshold) * h_scale
            if sig_arg >= 20.0:
                w_spatial = 1.0
            elif sig_arg <= -20.0:
                w_spatial = 0.0
            else:
                w_spatial = 1.0 / (1.0 + np.exp(-sig_arg))
            w_temporal = 1.0 - w_spatial

        # ---- 调用动态边构建 ----
        edge_src, edge_dst, edge_attrs, num_edges = _build_edges_dynamic_numba(
            node_offsets, valid_cube,
            lut_array, lut_lengths, lut_radius,
            w_spatial, w_temporal,
            spatial_boost, temporal_extend
        )

        return edge_src, edge_dst, edge_attrs, num_edges, H_value, w_spatial, w_temporal


# ============================================================
# 纯 Python 回退
# ============================================================

else:

    def _build_edges_dynamic_numba(
        node_offsets, valid_cube, lut_array, lut_lengths, lut_radius,
        w_spatial, w_temporal, spatial_boost, temporal_extend
    ):
        """纯 Python 回退：动态边构建"""
        N = len(node_offsets)
        R = lut_radius
        ct = valid_cube.shape[0] // 2
        ch = valid_cube.shape[1] // 2
        cw = valid_cube.shape[2] // 2
        ct_size = valid_cube.shape[0]
        ch_size = valid_cube.shape[1]
        cw_size = valid_cube.shape[2]

        # 构建偏移量查找表
        offset_to_idx = {}
        for i in range(N):
            key = (int(node_offsets[i, 0]), int(node_offsets[i, 1]), int(node_offsets[i, 2]))
            offset_to_idx[key] = i

        src_list, dst_list, attr_list = [], [], []

        for i in range(N):
            dt = int(node_offsets[i, 0])
            dh = int(node_offsets[i, 1])
            dw = int(node_offsets[i, 2])

            if dt == 0 and dh == 0 and dw == 0:
                continue

            lut_i = dt + R
            lut_j = dh + R
            lut_k = dw + R
            path_len = int(lut_lengths[lut_i, lut_j, lut_k])
            first_valid = -1

            if path_len > 0:
                effective_path_len = int(path_len * spatial_boost)
                for m in range(min(path_len, effective_path_len)):
                    pt = int(lut_array[lut_i, lut_j, lut_k, m, 0])
                    ph = int(lut_array[lut_i, lut_j, lut_k, m, 1])
                    pw = int(lut_array[lut_i, lut_j, lut_k, m, 2])
                    gt, gh, gw = pt + ct, ph + ch, pw + cw
                    if not (0 <= gt < ct_size and 0 <= gh < ch_size and 0 <= gw < cw_size):
                        break
                    if not valid_cube[gt, gh, gw]:
                        break
                    key = (pt, ph, pw)
                    if key in offset_to_idx:
                        first_valid = offset_to_idx[key]
                        break

            if first_valid >= 0:
                src_spatial = float(dh - node_offsets[first_valid, 1])
                src_spatial_w = float(dw - node_offsets[first_valid, 2])
                src_temporal = float(dt - node_offsets[first_valid, 0])

                attr = np.array([
                    w_spatial * src_temporal + w_temporal * src_temporal,
                    w_spatial * src_spatial,
                    w_spatial * src_spatial_w
                ], dtype=np.float32)
                src_list.append(i)
                dst_list.append(first_valid)
                attr_list.append(attr)
            else:
                attr = np.array([
                    float(dt),
                    w_spatial * float(dh),
                    w_spatial * float(dw)
                ], dtype=np.float32)
                src_list.append(i)
                dst_list.append(0)
                attr_list.append(attr)

            # 低异质性：时序扩展
            if w_temporal > 0.5 and temporal_extend > 0:
                for t_ext in range(1, temporal_extend + 1):
                    for sign in [1, -1]:
                        ext_dt = dt + sign * t_ext
                        key = (ext_dt, dh, dw)
                        if key in offset_to_idx:
                            ext_idx = offset_to_idx[key]
                            attr = np.array([
                                w_temporal * float(sign * t_ext), 0.0, 0.0
                            ], dtype=np.float32)
                            src_list.append(i)
                            dst_list.append(ext_idx)
                            attr_list.append(attr)

            # 高异质性：空间近邻
            if w_spatial > 0.5 and spatial_boost > 1.0:
                neighbors = [
                    (dt, dh + 1, dw), (dt, dh - 1, dw),
                    (dt, dh, dw + 1), (dt, dh, dw - 1),
                    (dt + 1, dh, dw), (dt - 1, dh, dw),
                ]
                for sndt, sndh, sndw in neighbors:
                    key = (sndt, sndh, sndw)
                    if key in offset_to_idx:
                        sn_idx = offset_to_idx[key]
                        if sn_idx != i:
                            # 检查重复
                            dup = any(s == i and d == sn_idx for s, d in zip(src_list, dst_list))
                            if not dup:
                                attr = np.array([
                                    w_spatial * float(sndt - dt),
                                    w_spatial * float(sndh - dh),
                                    w_spatial * float(sndw - dw)
                                ], dtype=np.float32)
                                src_list.append(i)
                                dst_list.append(sn_idx)
                                attr_list.append(attr)

        n = len(src_list)
        if n == 0:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                    np.zeros((0, 3), dtype=np.float32), 0)
        return (np.array(src_list, dtype=np.int64),
                np.array(dst_list, dtype=np.int64),
                np.array(attr_list, dtype=np.float32), n)

    def _build_edges_dynamic_with_heterogeneity(
        node_offsets, valid_cube, data_cube,
        lut_array, lut_lengths, lut_radius,
        h_threshold, h_scale, spatial_boost, temporal_extend, min_valid
    ):
        """纯 Python 回退：完整版（含异质性计算）"""
        from .heterogeneity_analyzer import (
            _compute_heterogeneity_single,
            _compute_spatial_temporal_weights
        )

        H_value = _compute_heterogeneity_single(data_cube, valid_cube, min_valid)
        w_spatial, w_temporal = _compute_spatial_temporal_weights(
            H_value, h_threshold, h_scale
        )

        edge_src, edge_dst, edge_attrs, num_edges = _build_edges_dynamic_numba(
            node_offsets, valid_cube, lut_array, lut_lengths, lut_radius,
            w_spatial, w_temporal, spatial_boost, temporal_extend
        )

        return edge_src, edge_dst, edge_attrs, num_edges, H_value, w_spatial, w_temporal


# ============================================================
# 高级接口类
# ============================================================

class DynamicEdgeBuilder:
    """
    动态边构建器（高级接口）。

    封装 Numba JIT 内核，提供便捷的 Python 接口。
    根据空间异质性指数动态调整边构建策略。

    使用示例：
        >>> builder = DynamicEdgeBuilder(
        ...     h_threshold=0.25, h_scale=10.0,
        ...     spatial_boost=1.5, temporal_extend=3
        ... )
        >>> src, dst, attrs, n = builder.build(
        ...     node_offsets, valid_cube, data_cube,
        ...     lut_array, lut_lengths, lut_radius,
        ...     w_spatial=0.8, w_temporal=0.2
        ... )
    """

    def __init__(
        self,
        h_threshold: float = 0.25,
        h_scale: float = 10.0,
        spatial_boost: float = 1.5,
        temporal_extend: int = 3,
        min_valid: int = 8,
    ):
        """
        Parameters
        ----------
        h_threshold : float
            异质性阈值
        h_scale : float
            sigmoid 缩放因子
        spatial_boost : float
            高异质性区域的路径放松系数
        temporal_extend : int
            低异质性区域的时序扩展步数
        min_valid : int
            最小有效像素数
        """
        self.h_threshold = h_threshold
        self.h_scale = h_scale
        self.spatial_boost = spatial_boost
        self.temporal_extend = temporal_extend
        self.min_valid = min_valid

    def build(
        self,
        node_offsets: np.ndarray,
        valid_cube: np.ndarray,
        lut_array: np.ndarray,
        lut_lengths: np.ndarray,
        lut_radius: int,
        w_spatial: float,
        w_temporal: float,
    ):
        """
        使用给定的时空权重构建动态边。

        Parameters
        ----------
        node_offsets : (N, 3) int32
        valid_cube : (ct_size, ch_size, cw_size) bool
        lut_array, lut_lengths, lut_radius : Bresenham 查找表
        w_spatial : float
        w_temporal : float

        Returns
        -------
        edge_src, edge_dst, edge_attrs, num_edges
        """
        return _build_edges_dynamic_numba(
            node_offsets, valid_cube,
            lut_array, lut_lengths, lut_radius,
            w_spatial, w_temporal,
            self.spatial_boost, self.temporal_extend
        )

    def build_with_heterogeneity(
        self,
        node_offsets: np.ndarray,
        valid_cube: np.ndarray,
        data_cube: np.ndarray,
        lut_array: np.ndarray,
        lut_lengths: np.ndarray,
        lut_radius: int,
    ):
        """
        自动计算异质性并构建动态边。

        内部调用 _build_edges_dynamic_with_heterogeneity，
        将异质性分析和边构建合并为一次 Numba JIT 调用。

        Returns
        -------
        edge_src, edge_dst, edge_attrs, num_edges : 边数据
        H_value : float - 异质性指数
        w_spatial : float - 空间权重
        w_temporal : float - 时序权重
        """
        return _build_edges_dynamic_with_heterogeneity(
            node_offsets, valid_cube, data_cube,
            lut_array, lut_lengths, lut_radius,
            self.h_threshold, self.h_scale,
            self.spatial_boost, self.temporal_extend,
            self.min_valid
        )
