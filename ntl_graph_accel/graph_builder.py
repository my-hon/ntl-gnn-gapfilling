"""
核心图构建模块
==============
与参考实现 build_dataset.py 的 graph_generate 精确对齐。

关键特性：
  1. assign_quadrants_plana 完整象限分配（含子情况边界条件）
  2. Round-robin 象限选择（非配额填充）
  3. 三种边类型：Bresenham 遮挡边 + 同时空辅助边 + 自环
  4. EXT_RANGE=6 + 自适应扩展
  5. 中心节点 index=0, 特征=[-1.0]
  6. 边构建与节点选择交错执行
  7. 输出为 dict 格式（与参考一致，含 ground_truth）
  8. 保留 GPU Bresenham 查找表和缓存复用（v1 特色）
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass

from .config import Config
from .gpu_bresenham import BresenhamLookupTable, GPUBresenhamAccelerator
from .graph_cache import GraphCache, GraphTemplate

logger = logging.getLogger(__name__)


@dataclass
class SubGraph:
    """
    单个子图数据结构。
    可直接序列化为Pickle用于GNN训练。
    """
    # 中心节点全局坐标 (t, h, w)
    center_pos: np.ndarray
    # 节点特征值 (N, 1) - 归一化后的NTL值
    node_features: np.ndarray
    # 边索引 (2, E) - [src, dst]
    edge_index: np.ndarray
    # 边属性 (E, 3) - 归一化后的3D偏移量
    edge_attr: np.ndarray
    # ground truth 标签 (N, 1)
    ground_truth: np.ndarray
    # 位置 (1, 3)
    position: np.ndarray
    # 图的节点数
    num_nodes: int

    def to_dict(self) -> Dict:
        """转为字典（便于序列化）"""
        return {
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "ground_truth": self.ground_truth,
            "position": self.position,
        }


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
# 辅助函数
# ============================================================


def bresenham_3d(start, end):
    """
    标准 3D Bresenham 直线算法。
    精确复制参考实现的逻辑。
    返回路径点列表，包含起点和终点。
    """
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

    return points


def get_blocks_through_line(data_array, start, center_pos):
    """
    精确复制参考实现的 get_blocks_through_line。
    获取从 start 到中心元素的连线穿过的所有有效方块（去掉首尾，过滤 NaN）。
    """
    line_points = bresenham_3d(start, center_pos)
    # 去掉起点和终点
    if len(line_points) <= 2:
        return []
    through_points = line_points[1:-1]
    # 过滤 NaN
    valid_points = []
    for pt in through_points:
        x, y, z = int(pt[0]), int(pt[1]), int(pt[2])
        if 0 <= x < data_array.shape[0] and \
           0 <= y < data_array.shape[1] and \
           0 <= z < data_array.shape[2]:
            if not np.isnan(data_array[x, y, z]):
                valid_points.append((x, y, z))
    return valid_points


def filter_triplets(node_pos_list, new_triplet):
    """
    精确复制参考实现的 filter_triplets。
    筛选满足条件的三元组：相同时间不同空间 + 相同空间不同时间。
    """
    x_new, y_new, z_new = new_triplet
    time_filtered = [
        triplet
        for triplet in node_pos_list
        if (triplet[0] == x_new and (triplet[1] != y_new or triplet[2] != z_new))
    ]
    spatial_filtered = [
        triplet
        for triplet in node_pos_list
        if ((triplet[1] == y_new or triplet[2] == z_new) and triplet[0] != x_new)
    ]
    return time_filtered + spatial_filtered


def sample_validate(data_array, search_node, quad_class_arr):
    """
    精确复制参考实现的 sample_validate。
    检查每个象限是否有足够的有效像素。
    """
    max_quad = int(quad_class_arr.max())
    if max_quad == 0:
        return False
    quad_lower_limit = np.full(max_quad, search_node // max_quad, dtype=np.uint8)
    quad_lower_limit[:search_node % max_quad] += 1
    for i in range(max_quad):
        cls_count = np.sum((quad_class_arr == i + 1) & ~np.isnan(data_array))
        if cls_count < quad_lower_limit[i]:
            return False
    return True


def compute_distances(data_array):
    """
    计算每个体素到中心的欧氏距离和偏移量。
    返回 (distances_arr, offset_arr)
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
                offset_arr[x, y, z] = [dt, dh, dw]
                distances_arr[x, y, z] = np.sqrt(dt**2 + dh**2 + dw**2)

    return distances_arr, offset_arr


def sort_indices_by_distance(distances_arr):
    """
    按距离排序返回索引列表 (N, 3)。
    """
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
    return [p[1] for p in pairs]


def graph_generate(data_array, search_node, distances_arr, offset_arr, quad_class_arr):
    """
    精确复制参考实现的 graph_generate。
    Round-robin 象限选择 + 三种边类型 + 交错构建。
    """
    sort_pos_list = sort_indices_by_distance(distances_arr)

    center_pos = (
        data_array.shape[0] // 2,
        data_array.shape[1] // 2,
        data_array.shape[2] // 2,
    )

    node_pos_list = [center_pos]
    node_feature_list = [[-1.0]]
    sparse_adj_mat = []
    edge_feature_list = []
    quad_cls_flag = 1

    while len(node_feature_list) < search_node + 1:
        for pos_now in sort_pos_list:
            if np.isnan(data_array[pos_now]):
                sort_pos_list.remove(pos_now)
                continue
            if quad_class_arr[pos_now] == quad_cls_flag:
                # Type A: Bresenham 遮挡边
                through_pos_list = get_blocks_through_line(
                    data_array, pos_now, center_pos
                )
                node_feature_list.append([data_array[pos_now]])
                node_pos_list.append(tuple(pos_now))

                if len(through_pos_list) == 0:
                    # 无遮挡，连接到中心
                    sparse_adj_mat.append([len(node_feature_list) - 1, 0])
                    edge_feature_list.append(tuple(offset_arr[pos_now]))
                else:
                    found = False
                    for through_pos in through_pos_list:
                        if through_pos in node_pos_list:
                            sparse_adj_mat.append([
                                len(node_feature_list) - 1,
                                node_pos_list.index(through_pos),
                            ])
                            offset_now = np.array(pos_now) - np.array(through_pos)
                            edge_feature_list.append(tuple(offset_now))
                            found = True
                            break
                    if not found:
                        # for-else: 遍历完未找到，连接到中心
                        sparse_adj_mat.append([len(node_feature_list) - 1, 0])
                        edge_feature_list.append(tuple(offset_arr[pos_now]))

                # Type B: 同时空辅助边
                same_stpos_list = filter_triplets(node_pos_list, pos_now)
                for pos_samest in same_stpos_list:
                    if distances_arr[pos_now] <= distances_arr[pos_samest]:
                        continue
                    sparse_conn_samest = [
                        len(node_feature_list) - 1,
                        node_pos_list.index(pos_samest),
                    ]
                    if sparse_conn_samest not in sparse_adj_mat:
                        sparse_adj_mat.append(sparse_conn_samest)
                        offset_samest = np.array(pos_now) - np.array(pos_samest)
                        edge_feature_list.append(tuple(offset_samest))

                # Type C: 自环
                sparse_adj_mat.append([len(node_feature_list) - 1, len(node_feature_list) - 1])
                edge_feature_list.append((0, 0, 0))

                sort_pos_list.remove(pos_now)
                quad_cls_flag = quad_cls_flag + 1
                if quad_cls_flag > 6:
                    quad_cls_flag = 1
                break

    ground_truth_list = [data_array[pos_now] for pos_now in node_pos_list]
    return (
        node_feature_list,
        node_pos_list,
        sparse_adj_mat,
        edge_feature_list,
        ground_truth_list,
    )


class GraphBuilder:
    """
    时空图构建器。

    核心算法与参考实现对齐：
    1. assign_quadrants_plana 完整象限分配
    2. round-robin 节点选择
    3. 三种边类型：Bresenham遮挡 + 辅助边 + 自环
    4. EXT_RANGE=6 + 自适应扩展
    5. 中心节点 index=0, 特征=[-1.0]
    6. 输出 dict 格式（含 ground_truth）

    v1 特色保留：
    - GPU Bresenham 查找表（可选加速）
    - LRU 缓存复用（可选）
    """

    def __init__(self, config: Config, data: np.ndarray):
        """
        Parameters
        ----------
        config : Config
            全局配置
        data : np.ndarray
            完整NTL数据 (T, H, W)
        """
        self.config = config
        self.data = data
        self.T, self.H, self.W = data.shape

        # 初始化 v1 加速组件（可选）
        self.cache: Optional[GraphCache] = None
        self.lookup_table: Optional[BresenhamLookupTable] = None
        self.gpu_accel: Optional[GPUBresenhamAccelerator] = None

        self._init_accelerators()

    def _init_accelerators(self):
        """初始化 v1 加速器（可选功能，不影响核心算法正确性）"""
        accel = self.config.accel

        # 缓存
        if accel.use_cache:
            self.cache = GraphCache(
                cache_dir=self.config.cache_dir,
                max_size=accel.cache_max_size,
                quantization_step=accel.cache_quantization
            )
            logger.info(f"缓存已启用, 量化步长={accel.cache_quantization}")

        # Bresenham查找表（可选，核心算法使用纯Python bresenham_3d）
        if accel.bresenham_lookup:
            self.lookup_table = BresenhamLookupTable(
                max_radius=self.config.graph.max_radius
            )
            lut_path = f"{self.config.cache_dir}/bresenham_lut.npz"
            import os
            if os.path.exists(lut_path):
                self.lookup_table.load(lut_path)
                logger.info("Bresenham查找表已从磁盘加载")
            else:
                self.lookup_table.build()
                self.lookup_table.save(lut_path)
                logger.info("Bresenham查找表已构建并保存")

        # GPU加速器（可选）
        if accel.use_cuda:
            try:
                self.gpu_accel = GPUBresenhamAccelerator(
                    max_radius=self.config.graph.max_radius,
                    max_path_len=self.config.graph.max_bresenham_len
                )
            except Exception as e:
                logger.warning(f"GPU加速器初始化失败: {e}, 将使用CPU模式")

    def build_single(self, tc: int, hc: int, wc: int) -> Optional[Dict[str, Any]]:
        """
        为单个位置构建子图。
        精确复制参考实现的 process_index 逻辑。
        返回 dict 格式（与参考一致）。

        Parameters
        ----------
        tc, hc, wc : int
            中心坐标

        Returns
        -------
        dict or None
            包含 node_features, edge_index, edge_attr, ground_truth, position
        """
        graph_cfg = self.config.graph
        search_node = graph_cfg.search_node
        ext_range = self.config.data.ext_range

        # 自适应扩展
        ext_amount = 0
        while True:
            split_data = self.data[
                tc - ext_range - ext_amount : tc + ext_range + 1 + ext_amount,
                hc - ext_range - ext_amount : hc + ext_range + 1 + ext_amount,
                wc - ext_range - ext_amount : wc + ext_range + 1 + ext_amount,
            ]

            # 检查边界
            if split_data.size == 0:
                ext_amount += 1
                if ext_amount > graph_cfg.max_radius:
                    return None
                continue

            # 计算象限分类
            quad_class_arr = assign_quadrants_plana(split_data)

            # 验证每个象限是否有足够的有效像素
            if sample_validate(split_data, search_node, quad_class_arr):
                break

            ext_amount += 1
            if ext_amount > graph_cfg.max_radius:
                return None

        # 计算距离和偏移
        distances_arr, offset_arr = compute_distances(split_data)

        # 生成图
        (
            node_feature_list,
            _,
            sparse_adj_mat,
            edge_feature_list,
            ground_truth_list,
        ) = graph_generate(
            split_data, search_node, distances_arr, offset_arr, quad_class_arr
        )

        ground_truth_arr = np.array(ground_truth_list, dtype=np.float32)
        ground_truth_arr = np.expand_dims(ground_truth_arr, axis=-1)

        graph_data = {
            "node_features": np.array(node_feature_list, dtype=np.float32),
            "edge_index": np.array(sparse_adj_mat, dtype=np.uint16).T,
            "edge_attr": np.array(edge_feature_list, dtype=np.float32) / self.config.data.edge_scale,
            "ground_truth": ground_truth_arr,
            "position": np.array([[tc, hc, wc]], dtype=np.uint16),
        }

        return graph_data

    def build_batch(self, positions: np.ndarray) -> List[Dict[str, Any]]:
        """
        批量构建子图。

        Parameters
        ----------
        positions : np.ndarray
            形状 (N, 3) 的坐标数组

        Returns
        -------
        List[Dict]
            成功构建的子图列表
        """
        graphs = []
        total = len(positions)

        for i, (tc, hc, wc) in enumerate(positions):
            graph = self.build_single(int(tc), int(hc), int(wc))
            if graph is not None:
                graphs.append(graph)

            if (i + 1) % 10000 == 0:
                logger.info(f"进度: {i+1}/{total}, 已构建: {len(graphs)}")

        return graphs
