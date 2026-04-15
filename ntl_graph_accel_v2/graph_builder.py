"""
核心图构建模块（v2）
====================
与参考实现 build_dataset.py 的 graph_generate 精确对齐。

关键改动：
  1. Round-robin 象限选择（非配额填充）
  2. 三种边类型：Bresenham 遮挡边 + 同时空辅助边 + 自环
  3. EXT_RANGE=6 + 自适应扩展
  4. 中心节点 index=0, 特征=[-1.0]
  5. 边构建与节点选择交错执行
  6. 输出为 dict 格式（与参考一致）
"""

import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from .config import Config
from .jit_kernels import (
    compute_distances_numba,
    assign_quadrants_numba,
    sort_indices_by_distance_numba,
    count_quadrant_valid_numba,
    bresenham_3d_numba,
    HAS_NUMBA,
)

logger = logging.getLogger(__name__)


@dataclass
class SubGraph:
    """子图数据结构（与 v1 兼容）"""
    center_pos: np.ndarray
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    ground_truth: np.ndarray
    position: np.ndarray
    num_nodes: int

    def to_dict(self):
        return {
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "ground_truth": self.ground_truth,
            "position": self.position,
        }


def _filter_triplets(node_pos_list, new_triplet):
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


def _get_blocks_through_line(data_array, start, center_pos):
    """
    精确复制参考实现的 get_blocks_through_line。
    获取从 start 到中心元素的连线穿过的所有有效方块（去掉首尾，过滤 NaN）。
    """
    line_points = bresenham_3d_numba(start, center_pos)
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


def _sample_validate(data_array, search_node, quad_class_arr):
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


def _graph_generate(data_array, search_node, distances_arr, offset_arr, quad_class_arr):
    """
    精确复制参考实现的 graph_generate。
    Round-robin 象限选择 + 三种边类型 + 交错构建。
    """
    sort_pos_list = sort_indices_by_distance_numba(distances_arr)

    # 将排序后的索引转为 Python list 以支持 remove 操作
    sort_pos_list = [tuple(int(v) for v in pos) for pos in sort_pos_list]

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
                through_pos_list = _get_blocks_through_line(
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
                same_stpos_list = _filter_triplets(node_pos_list, pos_now)
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
    时空图构建器（v2 - 与参考实现对齐版）。

    核心算法：
    - Round-robin 象限选择
    - 三种边类型（Bresenham + 辅助 + 自环）
    - EXT_RANGE=6 + 自适应扩展
    - 中心节点 index=0, 特征=[-1.0]
    """

    def __init__(self, config: Config, data: np.ndarray):
        self.config = config
        self.data = data
        self.T, self.H, self.W = data.shape

        # 预热 Numba JIT
        if config.accel.use_numba and HAS_NUMBA:
            logger.info("预热 Numba JIT 编译...")
            self._warmup_jit()
            logger.info("Numba JIT 编译完成")

    def _warmup_jit(self):
        """用小数据触发 Numba JIT 编译"""
        tiny = np.random.random((5, 5, 5)).astype(np.float32)
        compute_distances_numba(tiny)
        assign_quadrants_numba(tiny)
        sort_indices_by_distance_numba(np.zeros((5, 5, 5), dtype=np.float32))
        count_quadrant_valid_numba(
            np.zeros((5, 5, 5), dtype=np.uint8),
            np.zeros((5, 5, 5), dtype=np.bool_)
        )
        bresenham_3d_numba(np.array([0, 0, 0], dtype=np.int32),
                           np.array([2, 2, 2], dtype=np.int32))

    def build_single(self, tc: int, hc: int, wc: int) -> Optional[Dict[str, Any]]:
        """
        为单个位置构建子图。
        精确复制参考实现的 process_index 逻辑。
        返回 dict 格式（与参考一致）。
        """
        graph_cfg = self.config.graph
        search_node = graph_cfg.search_node
        ext_range = graph_cfg.ext_range

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
                if ext_amount > graph_cfg.max_ext:
                    return None
                continue

            # 计算象限分类
            quad_class_arr = assign_quadrants_numba(split_data)

            # 验证每个象限是否有足够的有效像素
            if _sample_validate(split_data, search_node, quad_class_arr):
                break

            ext_amount += 1
            if ext_amount > graph_cfg.max_ext:
                return None

        # 计算距离和偏移
        distances_arr, offset_arr = compute_distances_numba(split_data)

        # 生成图
        (
            node_feature_list,
            _,
            sparse_adj_mat,
            edge_feature_list,
            ground_truth_list,
        ) = _graph_generate(
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
        """批量构建子图"""
        graphs = []
        total = len(positions)
        for i, (tc, hc, wc) in enumerate(positions):
            graph = self.build_single(int(tc), int(hc), int(wc))
            if graph is not None:
                graphs.append(graph)
            if (i + 1) % 10000 == 0:
                logger.info(f"进度: {i+1}/{total}, 已构建: {len(graphs)}")
        return graphs

    def build_training_data(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """
        自然断点采样 + 并行构建训练数据。
        精确复制参考实现的 train_data_generate 逻辑。
        """
        graph_cfg = self.config.graph
        natural_breaks = graph_cfg.natural_breaks
        sample_pc = graph_cfg.sample_per_class
        search_node = graph_cfg.search_node

        edge_time = self.config.data.edge_time
        edge_height = self.config.data.edge_height
        edge_width = self.config.data.edge_width

        cut_data = data[
            edge_time:-edge_time, edge_height:-edge_height, edge_width:-edge_width
        ]

        all_graphs = []
        for i in range(len(natural_breaks) - 1):
            indices = np.array(np.where(
                (cut_data > natural_breaks[i]) & (cut_data < natural_breaks[i + 1])
            )).T + np.array([[edge_time, edge_height, edge_width]])

            if len(indices) == 0:
                logger.info(f"类别 {i+1}: 无有效像素")
                continue

            random_choice = np.random.choice(
                np.arange(len(indices)),
                size=sample_pc,
                replace=len(indices) < sample_pc,
            )
            selected_indices = indices[random_choice].tolist()

            logger.info(
                f"类别 {i+1}: 范围 [{natural_breaks[i]}, {natural_breaks[i+1]}], "
                f"有效像素: {len(indices)}, 采样: {len(selected_indices)}"
            )

            class_graphs = []
            for pos_now in selected_indices:
                graph = self.build_single(int(pos_now[0]), int(pos_now[1]), int(pos_now[2]))
                if graph is not None:
                    class_graphs.append(graph)

            all_graphs.append(class_graphs)
            logger.info(f"类别 {i+1}: 构建完成, 有效图: {len(class_graphs)}")

        return all_graphs

    def build_prediction_data(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """
        逐天 NaN 处理 + 构建预测数据。
        精确复制参考实现的 predict_data_generate 逻辑。
        """
        edge_time = self.config.data.edge_time
        edge_height = self.config.data.edge_height
        edge_width = self.config.data.edge_width

        cut_data = data[
            edge_time:-edge_time, edge_height:-edge_height, edge_width:-edge_width
        ]
        indices_nan = np.array(np.where(np.isnan(cut_data))).T

        all_day_graphs = []
        for day_num in range(0, data.shape[0] - 2 * edge_time):
            logger.info(f"Day: {day_num + 1}")
            indices_nan_day = indices_nan[
                np.where(indices_nan[:, 0] == day_num)
            ] + np.array([[edge_time, edge_height, edge_width]])

            day_graphs = []
            for pos_now in indices_nan_day:
                graph = self.build_single(int(pos_now[0]), int(pos_now[1]), int(pos_now[2]))
                if graph is not None:
                    day_graphs.append(graph)

            all_day_graphs.append(day_graphs)
            logger.info(f"Day {day_num + 1}: NaN 位置: {len(indices_nan_day)}, 有效图: {len(day_graphs)}")

        return all_day_graphs
