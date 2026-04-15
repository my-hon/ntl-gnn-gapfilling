"""
Numba JIT 核心计算内核（v2）
==============================
将 v1 中 Python 循环密集的计算编译为机器码。

关键改动：
  - _select_nodes: Python for 循环 + list append → Numba JIT（纯数组操作）
  - _build_edges:  Python for 循环 + dict 查找 → Numba JIT（预分配数组）
  - _bresenham_trace: 查找表 + Numba 路径过滤（消除 Python 循环）
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# 尝试导入 numba，不可用时回退到纯 Python
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
    logger.info(f"Numba 可用, 版本={numba.__version__}")
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba 不可用, 将使用纯 Python 回退模式")


# ============================================================
# Numba JIT 内核函数
# ============================================================

if HAS_NUMBA:

    @njit(cache=True)
    def _get_region_id(dt, dh, dw):
        """根据偏移方向确定区域编号 (0-5)"""
        adt, adh, adw = abs(dt), abs(dh), abs(dw)

        if adt >= adh and adt >= adw:
            primary = 0
            sign = dt
        elif adh >= adt and adh >= adw:
            primary = 1
            sign = dh
        else:
            primary = 2
            sign = dw

        return primary * 2 + (1 if sign < 0 else 0)

    @njit(cache=True)
    def _select_nodes_numba(
        valid_cube,      # (ct_size, ch_size, cw_size) bool
        cube_data,       # (ct_size, ch_size, cw_size) float32
        num_nodes,       # int
        num_regions,     # int
    ):
        """
        Numba 加速的节点选择。
        替代 v1 中的 Python for 循环 + list append + dict。

        Returns
        -------
        node_offsets : (N, 3) int32
        node_features : (N,) float32
        region_ids : (N,) int32
        actual_num_nodes : int
        """
        ct_size, ch_size, cw_size = valid_cube.shape
        ct = ct_size // 2
        ch = ch_size // 2
        cw = cw_size // 2

        # 第一遍：收集所有有效像素及其距离
        max_valid = ct_size * ch_size * cw_size
        valid_dt = np.empty(max_valid, dtype=np.int32)
        valid_dh = np.empty(max_valid, dtype=np.int32)
        valid_dw = np.empty(max_valid, dtype=np.int32)
        valid_dist = np.empty(max_valid, dtype=np.float32)
        valid_count = 0

        for t in range(ct_size):
            for h in range(ch_size):
                for w in range(cw_size):
                    if valid_cube[t, h, w]:
                        dt = t - ct
                        dh = h - ch
                        dw = w - cw
                        valid_dt[valid_count] = dt
                        valid_dh[valid_count] = dh
                        valid_dw[valid_count] = dw
                        valid_dist[valid_count] = np.sqrt(
                            float(dt * dt + dh * dh + dw * dw)
                        )
                        valid_count += 1

        if valid_count == 0:
            return (np.zeros((0, 3), dtype=np.int32),
                    np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=np.int32), 0)

        # 第二遍：按距离排序（插入排序，对小数组足够快且 Numba 友好）
        order = np.arange(valid_count, dtype=np.int32)
        for i in range(1, valid_count):
            key = order[i]
            j = i - 1
            while j >= 0 and valid_dist[order[j]] > valid_dist[key]:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key

        # 第三遍：按区域配额选择节点
        quotas = np.zeros(num_regions, dtype=np.int32)
        q, rem = divmod(num_nodes, num_regions)
        for i in range(num_regions):
            quotas[i] = q + (1 if i < rem else 0)

        region_counts = np.zeros(num_regions, dtype=np.int32)

        # 预分配输出
        out_offsets = np.zeros((num_nodes, 3), dtype=np.int32)
        out_features = np.zeros(num_nodes, dtype=np.float32)
        out_regions = np.zeros(num_nodes, dtype=np.int32)
        selected = 0

        # 严格配额阶段
        for idx in range(valid_count):
            if selected >= num_nodes:
                break
            i = order[idx]
            dt = valid_dt[i]
            dh = valid_dh[i]
            dw = valid_dw[i]
            rid = _get_region_id(dt, dh, dw)

            if region_counts[rid] < quotas[rid]:
                out_offsets[selected, 0] = dt
                out_offsets[selected, 1] = dh
                out_offsets[selected, 2] = dw
                out_features[selected] = cube_data[dt + ct, dh + ch, dw + cw]
                out_regions[selected] = rid
                region_counts[rid] += 1
                selected += 1

        # 放宽配额阶段（补满）
        if selected < num_nodes:
            for idx in range(valid_count):
                if selected >= num_nodes:
                    break
                i = order[idx]
                dt = valid_dt[i]
                dh = valid_dh[i]
                dw = valid_dw[i]
                # 检查是否已选
                already = False
                for k in range(selected):
                    if (out_offsets[k, 0] == dt and
                        out_offsets[k, 1] == dh and
                        out_offsets[k, 2] == dw):
                        already = True
                        break
                if not already:
                    out_offsets[selected, 0] = dt
                    out_offsets[selected, 1] = dh
                    out_offsets[selected, 2] = dw
                    out_features[selected] = cube_data[dt + ct, dh + ch, dw + cw]
                    out_regions[selected] = _get_region_id(dt, dh, dw)
                    selected += 1

        return (out_offsets[:selected],
                out_features[:selected],
                out_regions[:selected],
                selected)

    @njit(cache=True)
    def _build_edges_numba(
        node_offsets,    # (N, 3) int32
        valid_cube,      # (ct_size, ch_size, cw_size) bool
        lut_array,       # (2R+1, 2R+1, 2R+1, max_len, 3) int16
        lut_lengths,     # (2R+1, 2R+1, 2R+1) int16
        lut_radius,      # int
    ):
        """
        Numba 加速的边构建。
        替代 v1 中的 Python for 循环 + dict 查找。

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
        # 偏移范围: [-R, R]，用 (dt+R, dh+R, dw+R) 作为索引
        lut_size = 2 * R + 1
        offset_lut = np.full((lut_size, lut_size, lut_size), -1, dtype=np.int32)
        for i in range(N):
            dt = node_offsets[i, 0]
            dh = node_offsets[i, 1]
            dw = node_offsets[i, 2]
            if 0 <= dt + R < lut_size and 0 <= dh + R < lut_size and 0 <= dw + R < lut_size:
                offset_lut[dt + R, dh + R, dw + R] = i

        # 预分配边数组（最大边数 = N-1 个非中心节点，每个最多 1 条边 + 1 自环）
        max_edges = 2 * N
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

            # 查找 Bresenham 路径
            li = dt + R
            lj = dh + R
            lk = dw + R
            path_len = lut_lengths[li, lj, lk]

            first_valid = -1  # -1 表示未找到

            if path_len > 0:
                for m in range(path_len):
                    pt = int(lut_array[li, lj, lk, m, 0])
                    ph = int(lut_array[li, lj, lk, m, 1])
                    pw = int(lut_array[li, lj, lk, m, 2])

                    # 检查是否在有效范围内
                    gt = pt + ct
                    gh = ph + ch
                    gw = pw + cw
                    if gt < 0 or gt >= ct_size or gh < 0 or gh >= ch_size or gw < 0 or gw >= cw_size:
                        break  # 越界

                    if not valid_cube[gt, gh, gw]:
                        break  # 遇到 NaN，截断

                    # 检查该位置是否是已选节点
                    if 0 <= pt + R < lut_size and 0 <= ph + R < lut_size and 0 <= pw + R < lut_size:
                        node_idx = offset_lut[pt + R, ph + R, pw + R]
                        if node_idx >= 0:
                            first_valid = node_idx
                            break

            # 添加边
            if first_valid >= 0:
                edge_src[num_edges] = i
                edge_dst[num_edges] = first_valid
                # 边属性：当前节点偏移 - 目标节点偏移
                edge_attrs[num_edges, 0] = dt - node_offsets[first_valid, 0]
                edge_attrs[num_edges, 1] = dh - node_offsets[first_valid, 1]
                edge_attrs[num_edges, 2] = dw - node_offsets[first_valid, 2]
                num_edges += 1
            else:
                # 无遮挡，直接连向中心节点
                edge_src[num_edges] = i
                edge_dst[num_edges] = 0
                edge_attrs[num_edges, 0] = float(dt)
                edge_attrs[num_edges, 1] = float(dh)
                edge_attrs[num_edges, 2] = float(dw)
                num_edges += 1

        return (edge_src[:num_edges],
                edge_dst[:num_edges],
                edge_attrs[:num_edges],
                num_edges)

    @njit(cache=True)
    def _count_region_valid_numba(valid_cube, num_regions):
        """Numba 加速的区域有效像素计数"""
        ct_size, ch_size, cw_size = valid_cube.shape
        ct = ct_size // 2
        ch = ch_size // 2
        cw = cw_size // 2
        counts = np.zeros(num_regions, dtype=np.int32)

        for t in range(ct_size):
            for h in range(ch_size):
                for w in range(cw_size):
                    if valid_cube[t, h, w]:
                        rid = _get_region_id(t - ct, h - ch, w - cw)
                        counts[rid] += 1
        return counts


# ============================================================
# 纯 Python 回退（Numba 不可用时）
# ============================================================

else:

    def _get_region_id(dt, dh, dw):
        adt, adh, adw = abs(dt), abs(dh), abs(dw)
        if adt >= adh and adt >= adw:
            return 0 if dt >= 0 else 1
        elif adh >= adt and adh >= adw:
            return 2 if dh >= 0 else 3
        else:
            return 4 if dw >= 0 else 5

    def _select_nodes_numba(valid_cube, cube_data, num_nodes, num_regions):
        """纯 Python 回退（与 v1 逻辑一致）"""
        ct = valid_cube.shape[0] // 2
        ch = valid_cube.shape[1] // 2
        cw = valid_cube.shape[2] // 2

        valid_ts, valid_hs, valid_ws = np.where(valid_cube)
        if len(valid_ts) == 0:
            return (np.zeros((0, 3), dtype=np.int32),
                    np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=np.int32), 0)

        dt = valid_ts - ct
        dh = valid_hs - ch
        dw = valid_ws - cw
        distances = np.sqrt(dt**2 + dh**2 + dw**2)
        sort_idx = np.argsort(distances)

        q, rem = divmod(num_nodes, num_regions)
        quotas = [q + 1 if i < rem else q for i in range(num_regions)]
        region_counts = [0] * num_regions

        offsets, features, regions = [], [], []
        for idx in sort_idx:
            if len(offsets) >= num_nodes:
                break
            d = (int(valid_ts[idx] - ct), int(valid_hs[idx] - ch), int(valid_ws[idx] - cw))
            rid = _get_region_id(*d)
            if region_counts[rid] < quotas[rid]:
                offsets.append(d)
                features.append(cube_data[valid_ts[idx], valid_hs[idx], valid_ws[idx]])
                regions.append(rid)
                region_counts[rid] += 1

        if len(offsets) < num_nodes:
            for idx in sort_idx:
                if len(offsets) >= num_nodes:
                    break
                d = (int(valid_ts[idx] - ct), int(valid_hs[idx] - ch), int(valid_ws[idx] - cw))
                if d not in offsets:
                    offsets.append(d)
                    features.append(cube_data[valid_ts[idx], valid_hs[idx], valid_ws[idx]])
                    regions.append(_get_region_id(*d))

        n = len(offsets)
        return (np.array(offsets, dtype=np.int32),
                np.array(features, dtype=np.float32),
                np.array(regions, dtype=np.int32), n)

    def _build_edges_numba(node_offsets, valid_cube, lut_array, lut_lengths, lut_radius):
        """纯 Python 回退"""
        N = len(node_offsets)
        R = lut_radius
        ct = valid_cube.shape[0] // 2
        ch = valid_cube.shape[1] // 2
        cw = valid_cube.shape[2] // 2

        offset_to_idx = {}
        for i in range(N):
            offset_to_idx[tuple(node_offsets[i])] = i

        src_list, dst_list, attr_list = [], [], []
        for i in range(N):
            dt, dh, dw = int(node_offsets[i, 0]), int(node_offsets[i, 1]), int(node_offsets[i, 2])
            if dt == 0 and dh == 0 and dw == 0:
                continue

            lut_i = dt + R
            lut_j = dh + R
            lut_k = dw + R
            path_len = int(lut_lengths[lut_i, lut_j, lut_k])
            first_valid = -1

            if path_len > 0:
                for m in range(path_len):
                    pt = int(lut_array[lut_i, lut_j, lut_k, m, 0])
                    ph = int(lut_array[lut_i, lut_j, lut_k, m, 1])
                    pw = int(lut_array[lut_i, lut_j, lut_k, m, 2])
                    gt, gh, gw = pt + ct, ph + ch, pw + cw
                    if not (0 <= gt < valid_cube.shape[0] and
                            0 <= gh < valid_cube.shape[1] and
                            0 <= gw < valid_cube.shape[2]):
                        break
                    if not valid_cube[gt, gh, gw]:
                        break
                    key = (pt, ph, pw)
                    if key in offset_to_idx:
                        first_valid = offset_to_idx[key]
                        break

            if first_valid >= 0:
                src_list.append(i)
                dst_list.append(first_valid)
                attr_list.append(node_offsets[i] - node_offsets[first_valid])
            else:
                src_list.append(i)
                dst_list.append(0)
                attr_list.append(node_offsets[i].astype(np.float32))

        n = len(src_list)
        if n == 0:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                    np.zeros((0, 3), dtype=np.float32), 0)
        return (np.array(src_list, dtype=np.int64),
                np.array(dst_list, dtype=np.int64),
                np.array(attr_list, dtype=np.float32), n)

    def _count_region_valid_numba(valid_cube, num_regions):
        ct = valid_cube.shape[0] // 2
        ch = valid_cube.shape[1] // 2
        cw = valid_cube.shape[2] // 2
        counts = np.zeros(num_regions, dtype=np.int32)
        ts, hs, ws = np.where(valid_cube)
        for i in range(len(ts)):
            rid = _get_region_id(int(ts[i]) - ct, int(hs[i]) - ch, int(ws[i]) - cw)
            counts[rid] += 1
        return counts
