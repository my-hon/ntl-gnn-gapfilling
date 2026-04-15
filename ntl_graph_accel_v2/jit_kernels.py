"""
Numba JIT 核心计算内核（v2）
==============================
与参考实现 build_dataset.py 对齐。

关键改动：
  - assign_quadrants_plana: 精确复制参考实现的象限分配逻辑（含边界条件差异）
  - compute_distances: 计算每个体素到中心的欧氏距离
  - bresenham_3d: 精确复制参考实现的 Bresenham 3D 直线算法
  - 移除 v2 的配额选择（改为 Python 层的 round-robin 选择）
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
# 象限分配 - 精确复制参考实现 assign_quadrants_plana
# ============================================================
# 参考实现中各象限的边界条件（>= vs >）不完全一致：
#
# 象限1 (offset[0] < 0): 所有4个子情况均使用 >= 和 >=
# 象限2 (offset[0] > 0): 所有4个子情况均使用 >= 和 >=
# 象限3 (offset[1] < 0): 子情况1: >=, >=
#                          子情况2: >=, >
#                          子情况3: >, >=
#                          子情况4: >, >
# 象限4 (offset[1] > 0): 子情况1: >, >
#                          子情况2: >, >=
#                          子情况3: >=, >
#                          子情况4: >=, >=
# 象限5 (offset[2] < 0): 子情况1: >, >
#                          子情况2: >, >=
#                          子情况3: >=, >
#                          子情况4: >=, >=
# 象限6 (offset[2] > 0): 子情况1: >=, >=
#                          子情况2: >=, >
#                          子情况3: >, >=
#                          子情况4: >, >

def _assign_quadrants_plana_python(dt, dh, dw):
    """
    精确复制参考实现的象限分配逻辑。
    输入为偏移量 (dt, dh, dw)，返回象限编号 1-6，0 表示未分配。
    """
    adt = abs(dt)
    adh = abs(dh)
    adw = abs(dw)

    # 象限1: offset[0] < 0, 所有子情况 >=, >=
    if dt < 0:
        if adt >= adh and adt >= adw:
            return 1

    # 象限2: offset[0] > 0, 所有子情况 >=, >=
    if dt > 0:
        if adt >= adh and adt >= adw:
            return 2

    # 象限3: offset[1] < 0
    if dh < 0:
        if dt > 0 and dw > 0:
            # 子情况1: >=, >=
            if adh >= adt and adh >= adw:
                return 3
        elif dt > 0 and dw <= 0:
            # 子情况2: >=, >
            if adh >= adt and adh > adw:
                return 3
        elif dt <= 0 and dw > 0:
            # 子情况3: >, >=
            if adh > adt and adh >= adw:
                return 3
        elif dt <= 0 and dw <= 0:
            # 子情况4: >, >
            if adh > adt and adh > adw:
                return 3

    # 象限4: offset[1] > 0
    if dh > 0:
        if dt > 0 and dw > 0:
            # 子情况1: >, >
            if adh > adt and adh > adw:
                return 4
        elif dt > 0 and dw <= 0:
            # 子情况2: >, >=
            if adh > adt and adh >= adw:
                return 4
        elif dt <= 0 and dw > 0:
            # 子情况3: >=, >
            if adh >= adt and adh > adw:
                return 4
        elif dt <= 0 and dw <= 0:
            # 子情况4: >=, >=
            if adh >= adt and adh >= adw:
                return 4

    # 象限5: offset[2] < 0
    if dw < 0:
        if dt > 0 and dh > 0:
            # 子情况1: >, >
            if adw > adt and adw > adh:
                return 5
        elif dt > 0 and dh <= 0:
            # 子情况2: >, >=
            if adw > adt and adw >= adh:
                return 5
        elif dt <= 0 and dh > 0:
            # 子情况3: >=, >
            if adw >= adt and adw > adh:
                return 5
        elif dt <= 0 and dh <= 0:
            # 子情况4: >=, >=
            if adw >= adt and adw >= adh:
                return 5

    # 象限6: offset[2] > 0
    if dw > 0:
        if dt > 0 and dh > 0:
            # 子情况1: >=, >=
            if adw >= adt and adw >= adh:
                return 6
        elif dt > 0 and dh <= 0:
            # 子情况2: >=, >
            if adw >= adt and adw > adh:
                return 6
        elif dt <= 0 and dh > 0:
            # 子情况3: >, >=
            if adw > adt and adw >= adh:
                return 6
        elif dt <= 0 and dh <= 0:
            # 子情况4: >, >
            if adw > adt and adw > adh:
                return 6

    return 0


def assign_quadrants_plana(data_array):
    """
    对整个子立方体计算象限分类数组。
    返回与 data_array 同形状的 uint8 数组，值 0=未分配, 1-6=象限。
    中心点 (0,0,0) 偏移保持为 0。
    """
    ct = data_array.shape[0] // 2
    ch = data_array.shape[1] // 2
    cw = data_array.shape[2] // 2
    quad_class_arr = np.zeros(data_array.shape, dtype=np.uint8)

    for x in range(data_array.shape[0]):
        for y in range(data_array.shape[1]):
            for z in range(data_array.shape[2]):
                dt = x - ct
                dh = y - ch
                dw = z - cw
                if dt == 0 and dh == 0 and dw == 0:
                    continue
                quad_class_arr[x, y, z] = _assign_quadrants_plana_python(dt, dh, dw)

    return quad_class_arr


# ============================================================
# Numba JIT 加速版本
# ============================================================

if HAS_NUMBA:

    @njit(cache=True)
    def _assign_quadrant_single(dt, dh, dw):
        """
        Numba 版本的象限分配，精确复制参考实现边界条件。
        返回 1-6，0 表示未分配。
        """
        adt = abs(dt)
        adh = abs(dh)
        adw = abs(dw)

        # 象限1: dt < 0, 所有子情况 >=, >=
        if dt < 0:
            if adt >= adh and adt >= adw:
                return 1

        # 象限2: dt > 0, 所有子情况 >=, >=
        if dt > 0:
            if adt >= adh and adt >= adw:
                return 2

        # 象限3: dh < 0
        if dh < 0:
            if dt > 0 and dw > 0:
                if adh >= adt and adh >= adw:
                    return 3
            elif dt > 0 and dw <= 0:
                if adh >= adt and adh > adw:
                    return 3
            elif dt <= 0 and dw > 0:
                if adh > adt and adh >= adw:
                    return 3
            elif dt <= 0 and dw <= 0:
                if adh > adt and adh > adw:
                    return 3

        # 象限4: dh > 0
        if dh > 0:
            if dt > 0 and dw > 0:
                if adh > adt and adh > adw:
                    return 4
            elif dt > 0 and dw <= 0:
                if adh > adt and adh >= adw:
                    return 4
            elif dt <= 0 and dw > 0:
                if adh >= adt and adh > adw:
                    return 4
            elif dt <= 0 and dw <= 0:
                if adh >= adt and adh >= adw:
                    return 4

        # 象限5: dw < 0
        if dw < 0:
            if dt > 0 and dh > 0:
                if adw > adt and adw > adh:
                    return 5
            elif dt > 0 and dh <= 0:
                if adw > adt and adw >= adh:
                    return 5
            elif dt <= 0 and dh > 0:
                if adw >= adt and adw > adh:
                    return 5
            elif dt <= 0 and dh <= 0:
                if adw >= adt and adw >= adh:
                    return 5

        # 象限6: dw > 0
        if dw > 0:
            if dt > 0 and dh > 0:
                if adw >= adt and adw >= adh:
                    return 6
            elif dt > 0 and dh <= 0:
                if adw >= adt and adw > adh:
                    return 6
            elif dt <= 0 and dh > 0:
                if adw > adt and adw >= adh:
                    return 6
            elif dt <= 0 and dh <= 0:
                if adw > adt and adw > adh:
                    return 6

        return 0

    @njit(cache=True)
    def compute_distances_numba(data_array):
        """
        Numba 加速的距离计算。
        返回 (distances_arr, offset_arr)
        distances_arr: (T,H,W) float32 - 每个体素到中心的欧氏距离
        offset_arr: (T,H,W,3) int8 - 每个体素到中心的偏移量
        """
        ct = data_array.shape[0] // 2
        ch = data_array.shape[1] // 2
        cw = data_array.shape[2] // 2
        T, H, W = data_array.shape

        distances_arr = np.zeros((T, H, W), dtype=np.float32)
        offset_arr = np.zeros((T, H, W, 3), dtype=np.int8)

        for x in range(T):
            for y in range(H):
                for z in range(W):
                    dt = x - ct
                    dh = y - ch
                    dw = z - cw
                    offset_arr[x, y, z, 0] = dt
                    offset_arr[x, y, z, 1] = dh
                    offset_arr[x, y, z, 2] = dw
                    distances_arr[x, y, z] = np.sqrt(
                        float(dt * dt + dh * dh + dw * dw)
                    )
        return distances_arr, offset_arr

    @njit(cache=True)
    def assign_quadrants_numba(data_array):
        """
        Numba 加速的象限分类数组计算。
        返回 (T,H,W) uint8 数组。
        """
        ct = data_array.shape[0] // 2
        ch = data_array.shape[1] // 2
        cw = data_array.shape[2] // 2
        T, H, W = data_array.shape
        quad_class_arr = np.zeros((T, H, W), dtype=np.uint8)

        for x in range(T):
            for y in range(H):
                for z in range(W):
                    dt = x - ct
                    dh = y - ch
                    dw = z - cw
                    if dt == 0 and dh == 0 and dw == 0:
                        continue
                    quad_class_arr[x, y, z] = _assign_quadrant_single(dt, dh, dw)

        return quad_class_arr

    @njit(cache=True)
    def sort_indices_by_distance_numba(distances_arr):
        """
        Numba 加速的距离排序。
        返回按距离排序的索引列表 (N, 3) int32。
        """
        T, H, W = distances_arr.shape
        total = T * H * W

        # 展平
        flat_dist = np.empty(total, dtype=np.float32)
        flat_x = np.empty(total, dtype=np.int32)
        flat_y = np.empty(total, dtype=np.int32)
        flat_z = np.empty(total, dtype=np.int32)

        idx = 0
        for x in range(T):
            for y in range(H):
                for z in range(W):
                    flat_dist[idx] = distances_arr[x, y, z]
                    flat_x[idx] = x
                    flat_y[idx] = y
                    flat_z[idx] = z
                    idx += 1

        # 插入排序（对小数组足够快且 Numba 友好）
        order = np.arange(total, dtype=np.int32)
        for i in range(1, total):
            key = order[i]
            j = i - 1
            while j >= 0 and flat_dist[order[j]] > flat_dist[key]:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key

        sorted_indices = np.empty((total, 3), dtype=np.int32)
        for i in range(total):
            sorted_indices[i, 0] = flat_x[order[i]]
            sorted_indices[i, 1] = flat_y[order[i]]
            sorted_indices[i, 2] = flat_z[order[i]]

        return sorted_indices

    @njit(cache=True)
    def count_quadrant_valid_numba(quad_class_arr, data_is_nan):
        """
        统计每个象限中非 NaN 的有效像素数量。
        返回 (6,) int32 数组，索引 0 对应象限 1。
        """
        counts = np.zeros(6, dtype=np.int32)
        T, H, W = quad_class_arr.shape
        for x in range(T):
            for y in range(H):
                for z in range(W):
                    q = quad_class_arr[x, y, z]
                    if q > 0 and not data_is_nan[x, y, z]:
                        counts[q - 1] += 1
        return counts

    @njit(cache=True)
    def bresenham_3d_numba(start, end):
        """
        Numba 版本的 3D Bresenham 直线算法。
        精确复制参考实现的逻辑。
        返回路径点数组 (N, 3) int32，包含起点和终点。
        """
        x1, y1, z1 = int(start[0]), int(start[1]), int(start[2])
        x2, y2, z2 = int(end[0]), int(end[1]), int(end[2])

        # 预分配最大可能长度
        max_len = max(abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)) + 1
        points = np.empty((max_len, 3), dtype=np.int32)
        points[0] = [x1, y1, z1]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1

        count = 1

        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while x1 != x2:
                x1 += xs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
                points[count] = [x1, y1, z1]
                count += 1
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while y1 != y2:
                y1 += ys
                if p1 >= 0:
                    x1 += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
                points[count] = [x1, y1, z1]
                count += 1
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while z1 != z2:
                z1 += zs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    x1 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
                points[count] = [x1, y1, z1]
                count += 1

        return points[:count]


# ============================================================
# 纯 Python 回退（Numba 不可用时）
# ============================================================

else:

    def compute_distances_numba(data_array):
        """纯 Python 回退的距离计算"""
        ct = data_array.shape[0] // 2
        ch = data_array.shape[1] // 2
        cw = data_array.shape[2] // 2
        T, H, W = data_array.shape

        distances_arr = np.zeros((T, H, W), dtype=np.float32)
        offset_arr = np.zeros((T, H, W, 3), dtype=np.int8)

        for x in range(T):
            for y in range(H):
                for z in range(W):
                    dt = x - ct
                    dh = y - ch
                    dw = z - cw
                    offset_arr[x, y, z] = [dt, dh, dw]
                    distances_arr[x, y, z] = np.sqrt(dt**2 + dh**2 + dw**2)
        return distances_arr, offset_arr

    def assign_quadrants_numba(data_array):
        """纯 Python 回退的象限分类"""
        return assign_quadrants_plana(data_array)

    def sort_indices_by_distance_numba(distances_arr):
        """纯 Python 回退的距离排序"""
        T, H, W = distances_arr.shape
        indices = []
        dists = []
        for x in range(T):
            for y in range(H):
                for z in range(W):
                    dists.append(distances_arr[x, y, z])
                    indices.append((x, y, z))
        pairs = list(zip(dists, indices))
        pairs.sort(key=lambda p: p[0])
        return np.array([p[1] for p in pairs], dtype=np.int32)

    def count_quadrant_valid_numba(quad_class_arr, data_is_nan):
        """纯 Python 回退的象限计数"""
        counts = np.zeros(6, dtype=np.int32)
        T, H, W = quad_class_arr.shape
        for x in range(T):
            for y in range(H):
                for z in range(W):
                    q = quad_class_arr[x, y, z]
                    if q > 0 and not data_is_nan[x, y, z]:
                        counts[q - 1] += 1
        return counts

    def bresenham_3d_numba(start, end):
        """纯 Python 回退的 Bresenham 3D"""
        x1, y1, z1 = int(start[0]), int(start[1]), int(start[2])
        x2, y2, z2 = int(end[0]), int(end[1]), int(end[2])
        points = [(x1, y1, z1)]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1

        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while x1 != x2:
                x1 += xs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
                points.append((x1, y1, z1))
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while y1 != y2:
                y1 += ys
                if p1 >= 0:
                    x1 += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
                points.append((x1, y1, z1))
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while z1 != z2:
                z1 += zs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    x1 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
                points.append((x1, y1, z1))
        return np.array(points, dtype=np.int32)
