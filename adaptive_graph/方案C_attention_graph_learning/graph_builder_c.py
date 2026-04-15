"""
方案C：图构建器（候选池 + MLP 节点选择）
==========================================
图构建阶段的实现，生成候选节点池并使用 MLP 进行节点选择。

与 v2 GraphBuilder 的区别：
  - v2: 固定选择 36 个节点，使用距离排序 + 区域配额
  - 方案C: 生成 64 个候选节点，使用 MLP 计算重要性分数选择 top-K

图构建阶段仍可使用 Numba 加速（候选节点收集），
但节点选择部分使用 MLP（learnable_graph.py 中的 LearnableNodeSelector）。
"""

import numpy as np
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .config_c import ConfigC, GraphBuildConfigC

logger = logging.getLogger(__name__)

# 尝试导入 Numba
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba 不可用, 将使用纯 Python 回退模式")


# ============================================================
# Numba JIT 内核（候选节点收集）
# ============================================================

if HAS_NUMBA:

    @njit(cache=True)
    def _get_region_id_c(dt, dh, dw):
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
    def _collect_candidate_nodes_numba(
        valid_cube,       # (ct_size, ch_size, cw_size) bool
        cube_data,        # (ct_size, ch_size, cw_size) float32
        candidate_size,   # int - 候选池大小
        num_regions,      # int - 区域数量
    ):
        """
        Numba 加速的候选节点收集。

        与 v2 _select_nodes_numba 的区别：
          - v2: 精确选择 num_nodes 个节点
          - 方案C: 收集 candidate_size 个候选节点（宽松选择）

        Returns
        -------
        node_offsets : (C, 3) int32 - 候选节点偏移
        node_features : (C,) float32 - 候选节点特征值
        region_ids : (C,) int32 - 区域编号
        actual_count : int - 实际收集的候选数量
        """
        ct_size, ch_size, cw_size = valid_cube.shape
        ct = ct_size // 2
        ch = ch_size // 2
        cw = cw_size // 2

        # 第一遍：收集所有有效像素
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

        # 第二遍：按距离排序
        order = np.arange(valid_count, dtype=np.int32)
        for i in range(1, valid_count):
            key = order[i]
            j = i - 1
            while j >= 0 and valid_dist[order[j]] > valid_dist[key]:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key

        # 第三遍：收集候选节点（宽松配额，允许超出）
        # 每个区域的候选配额 = candidate_size / num_regions * 1.5（宽松）
        loose_quotas = np.zeros(num_regions, dtype=np.int32)
        q, rem = divmod(candidate_size, num_regions)
        for i in range(num_regions):
            loose_quotas[i] = int(q * 1.5) + (2 if i < rem else 0)

        region_counts = np.zeros(num_regions, dtype=np.int32)

        # 预分配输出（候选池大小）
        out_offsets = np.zeros((candidate_size, 3), dtype=np.int32)
        out_features = np.zeros(candidate_size, dtype=np.float32)
        out_regions = np.zeros(candidate_size, dtype=np.int32)
        selected = 0

        # 严格配额阶段
        for idx in range(valid_count):
            if selected >= candidate_size:
                break
            i = order[idx]
            dt = valid_dt[i]
            dh = valid_dh[i]
            dw = valid_dw[i]
            rid = _get_region_id_c(dt, dh, dw)

            if region_counts[rid] < loose_quotas[rid]:
                out_offsets[selected, 0] = dt
                out_offsets[selected, 1] = dh
                out_offsets[selected, 2] = dw
                out_features[selected] = cube_data[dt + ct, dh + ch, dw + cw]
                out_regions[selected] = rid
                region_counts[rid] += 1
                selected += 1

        # 补满阶段
        if selected < candidate_size:
            for idx in range(valid_count):
                if selected >= candidate_size:
                    break
                i = order[idx]
                dt = valid_dt[i]
                dh = valid_dh[i]
                dw = valid_dw[i]
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
                    out_regions[selected] = _get_region_id_c(dt, dh, dw)
                    selected += 1

        return (out_offsets[:selected],
                out_features[:selected],
                out_regions[:selected],
                selected)

    @njit(cache=True)
    def _count_region_valid_c(valid_cube, num_regions):
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
                        rid = _get_region_id_c(t - ct, h - ch, w - cw)
                        counts[rid] += 1
        return counts


# ============================================================
# 纯 Python 回退
# ============================================================

else:

    def _get_region_id_c(dt, dh, dw):
        adt, adh, adw = abs(dt), abs(dh), abs(dw)
        if adt >= adh and adt >= adw:
            return 0 if dt >= 0 else 1
        elif adh >= adt and adh >= adw:
            return 2 if dh >= 0 else 3
        else:
            return 4 if dw >= 0 else 5

    def _collect_candidate_nodes_numba(valid_cube, cube_data, candidate_size, num_regions):
        """纯 Python 回退"""
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

        # 宽松配额
        q, rem = divmod(candidate_size, num_regions)
        loose_quotas = [int(q * 1.5) + (2 if i < rem else 0) for i in range(num_regions)]
        region_counts = [0] * num_regions

        offsets, features, regions = [], [], []
        for idx in sort_idx:
            if len(offsets) >= candidate_size:
                break
            d = (int(valid_ts[idx] - ct), int(valid_hs[idx] - ch), int(valid_ws[idx] - cw))
            rid = _get_region_id_c(*d)
            if region_counts[rid] < loose_quotas[rid]:
                offsets.append(d)
                features.append(cube_data[valid_ts[idx], valid_hs[idx], valid_ws[idx]])
                regions.append(rid)
                region_counts[rid] += 1

        # 补满
        if len(offsets) < candidate_size:
            for idx in sort_idx:
                if len(offsets) >= candidate_size:
                    break
                d = (int(valid_ts[idx] - ct), int(valid_hs[idx] - ch), int(valid_ws[idx] - cw))
                if d not in offsets:
                    offsets.append(d)
                    features.append(cube_data[valid_ts[idx], valid_hs[idx], valid_ws[idx]])
                    regions.append(_get_region_id_c(*d))

        n = len(offsets)
        return (np.array(offsets, dtype=np.int32),
                np.array(features, dtype=np.float32),
                np.array(regions, dtype=np.int32), n)

    def _count_region_valid_c(valid_cube, num_regions):
        ct = valid_cube.shape[0] // 2
        ch = valid_cube.shape[1] // 2
        cw = valid_cube.shape[2] // 2
        counts = np.zeros(num_regions, dtype=np.int32)
        ts, hs, ws = np.where(valid_cube)
        for i in range(len(ts)):
            rid = _get_region_id_c(int(ts[i]) - ct, int(hs[i]) - ch, int(ws[i]) - cw)
            counts[rid] += 1
        return counts


# ============================================================
# SubGraph 数据结构（与 v2 兼容）
# ============================================================

@dataclass
class SubGraphC:
    """
    方案C 子图数据结构。

    与 v2 SubGraph 兼容，额外增加候选池信息供训练阶段使用。
    """
    center_pos: np.ndarray
    node_features: np.ndarray
    edge_index_src: np.ndarray
    edge_index_dst: np.ndarray
    edge_attrs: np.ndarray
    center_value: float
    num_nodes: int

    # 方案C 新增字段
    candidate_offsets: Optional[np.ndarray] = None      # 候选节点偏移 (C, 3)
    candidate_features: Optional[np.ndarray] = None     # 候选节点特征 (C,)
    candidate_mask: Optional[np.ndarray] = None         # 候选有效掩码 (C,)
    candidate_regions: Optional[np.ndarray] = None      # 候选区域编号 (C,)

    def to_dict(self):
        """转换为字典格式（与 v2 兼容）"""
        result = {
            'center_pos': self.center_pos,
            'node_features': self.node_features,
            'edge_index': np.stack([self.edge_index_src, self.edge_index_dst], axis=0),
            'edge_attrs': self.edge_attrs,
            'center_value': self.center_value,
            'num_nodes': self.num_nodes
        }
        # 方案C 扩展字段
        if self.candidate_offsets is not None:
            result['candidate_offsets'] = self.candidate_offsets
            result['candidate_features'] = self.candidate_features
            result['candidate_mask'] = self.candidate_mask
            result['candidate_regions'] = self.candidate_regions
        return result

    @classmethod
    def from_v2_subgraph(cls, sg) -> 'SubGraphC':
        """从 v2 SubGraph 转换（兼容性接口）"""
        return cls(
            center_pos=sg.center_pos,
            node_features=sg.node_features,
            edge_index_src=sg.edge_index_src,
            edge_index_dst=sg.edge_index_dst,
            edge_attrs=sg.edge_attrs,
            center_value=sg.center_value,
            num_nodes=sg.num_nodes,
        )


# ============================================================
# GraphBuilderC - 方案C 图构建器
# ============================================================

class GraphBuilderC:
    """
    方案C 图构建器。

    工作流程：
      1. 自适应窗口扩展（与 v2 相同）
      2. 收集候选节点池（candidate_pool_size 个，使用 Numba 加速）
      3. 使用 MLP 节点选择器选择 top-K 节点（可选，图构建阶段可用简单选择）
      4. 构建初始边（基于距离的 KNN，训练阶段由 LearnableEdgeBuilder 优化）

    注意：图构建阶段生成的边是初始边（用于预训练/初始化），
          训练阶段会使用 LearnableEdgeBuilder 重新学习边权重。
    """

    def __init__(self, config: ConfigC, data: np.ndarray, valid_mask: np.ndarray):
        self.config = config
        self.data = data
        self.valid_mask = valid_mask
        self.T, self.H, self.W = data.shape
        self.graph_cfg = config.graph_build

        # MLP 节点选择器（图构建阶段使用）
        self.mlp_selector = None
        if self.graph_cfg.use_mlp_selector:
            self._init_mlp_selector()

        # 预热 Numba JIT
        if self.graph_cfg.use_numba and HAS_NUMBA:
            logger.info("预热 Numba JIT 编译...")
            self._warmup_jit()
            logger.info("Numba JIT 编译完成")

    def _init_mlp_selector(self):
        """初始化 MLP 节点选择器（用于图构建阶段）"""
        import torch
        from .learnable_graph import LearnableNodeSelector

        self.mlp_selector = LearnableNodeSelector(
            input_dim=self.graph_cfg.mlp_hidden_dim,
            hidden_dim=self.graph_cfg.mlp_hidden_dim,
            output_dim=16,
            num_layers=self.graph_cfg.mlp_num_layers,
            top_k=self.graph_cfg.num_nodes,
        )
        # 设为评估模式
        self.mlp_selector.eval()

    def _warmup_jit(self):
        """用小数据触发 Numba JIT 编译"""
        tiny_valid = np.ones((5, 5, 5), dtype=np.bool_)
        tiny_data = np.random.rand(5, 5, 5).astype(np.float32)
        _collect_candidate_nodes_numba(tiny_valid, tiny_data, 10, 6)

    def build_single(self, tc: int, hc: int, wc: int) -> Optional[SubGraphC]:
        """
        为单个位置构建子图。

        参数
        ----
        tc, hc, wc : int
            中心位置的时间、高度、宽度坐标

        返回
        ----
        SubGraphC 或 None
            构建成功返回子图，否则返回 None
        """
        graph_cfg = self.graph_cfg

        # 自适应窗口扩展
        radius = graph_cfg.initial_radius
        cube, valid_cube = self._crop_cube(tc, hc, wc, radius)

        while True:
            counts = _count_region_valid_c(valid_cube, graph_cfg.num_regions)
            # 候选池需要更多有效像素
            q, rem = divmod(graph_cfg.candidate_pool_size, graph_cfg.num_regions)
            min_required = q + (1 if rem > 0 else 0)

            if counts.min() >= min_required:
                break

            radius += 1
            if radius > graph_cfg.max_radius:
                return None
            cube, valid_cube = self._crop_cube(tc, hc, wc, radius)

        # ---- 收集候选节点池 ----
        offsets, features, regions, actual_count = _collect_candidate_nodes_numba(
            valid_cube, cube,
            graph_cfg.candidate_pool_size,
            graph_cfg.num_regions
        )
        if actual_count < 2:
            return None

        # ---- 节点选择 ----
        if self.mlp_selector is not None:
            selected_offsets, selected_features = self._mlp_select_nodes(
                offsets, features, regions, actual_count
            )
        else:
            # 简单选择：取前 num_nodes 个（按距离排序的）
            k = min(graph_cfg.num_nodes, actual_count)
            selected_offsets = offsets[:k]
            selected_features = features[:k]

        num_selected = len(selected_offsets)
        if num_selected < 2:
            return None

        # ---- 构建初始边（基于距离的 KNN）----
        edge_src, edge_dst, edge_attrs = self._build_initial_edges(
            selected_offsets, valid_cube
        )

        # ---- 归一化 ----
        selected_features = selected_features / self.config.data.feature_scale
        edge_attrs = edge_attrs / self.config.data.edge_scale

        center_value = float(self.data[tc, hc, wc])

        # ---- 构建候选掩码 ----
        candidate_mask = np.ones(actual_count, dtype=np.bool_)

        subgraph = SubGraphC(
            center_pos=np.array([tc, hc, wc], dtype=np.int32),
            node_features=selected_features.astype(np.float32),
            edge_index_src=edge_src.astype(np.int64),
            edge_index_dst=edge_dst.astype(np.int64),
            edge_attrs=edge_attrs.astype(np.float32),
            center_value=center_value,
            num_nodes=num_selected,
            # 方案C 扩展字段
            candidate_offsets=offsets.copy(),
            candidate_features=features.copy(),
            candidate_mask=candidate_mask,
            candidate_regions=regions.copy(),
        )

        return subgraph

    def build_batch(self, positions: np.ndarray) -> List[SubGraphC]:
        """
        批量构建子图。

        参数
        ----
        positions : np.ndarray
            位置数组，形状 (N, 3)，每行为 (t, h, w)

        返回
        ----
        List[SubGraphC]
            构建的子图列表
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

    def _mlp_select_nodes(
        self,
        offsets: np.ndarray,
        features: np.ndarray,
        regions: np.ndarray,
        actual_count: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 MLP 节点选择器选择 top-K 节点。

        参数
        ----
        offsets : np.ndarray
            候选节点偏移，形状 (C, 3)
        features : np.ndarray
            候选节点特征，形状 (C,)
        regions : np.ndarray
            候选区域编号，形状 (C,)
        actual_count : int
            实际候选数量

        返回
        ----
        selected_offsets : np.ndarray
            选中的节点偏移，形状 (K, 3)
        selected_features : np.ndarray
            选中的节点特征，形状 (K,)
        """
        import torch

        # 构造 MLP 输入: (dt, dh, dw, value)
        mlp_input = np.zeros((actual_count, 4), dtype=np.float32)
        mlp_input[:, :3] = offsets[:actual_count].astype(np.float32)
        mlp_input[:, 3] = features[:actual_count]

        # 转换为 PyTorch 张量
        input_tensor = torch.from_numpy(mlp_input).unsqueeze(0)  # (1, C, 4)
        offset_tensor = torch.from_numpy(
            offsets[:actual_count].astype(np.float32)
        ).unsqueeze(0)  # (1, C, 3)
        mask_tensor = torch.ones(1, actual_count, dtype=torch.bool)

        with torch.no_grad():
            indices, scores, _ = self.mlp_selector(
                input_tensor, offset_tensor, mask_tensor
            )

        # 转换回 numpy
        selected_idx = indices[0].cpu().numpy()
        selected_offsets = offsets[selected_idx]
        selected_features = features[selected_idx]

        return selected_offsets, selected_features

    def _build_initial_edges(
        self,
        node_offsets: np.ndarray,
        valid_cube: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构建初始边（基于距离的 KNN）。

        这是图构建阶段的初始边，训练阶段会被 LearnableEdgeBuilder 替代。

        参数
        ----
        node_offsets : np.ndarray
            节点偏移，形状 (N, 3)
        valid_cube : np.ndarray
            有效掩码立方体

        返回
        ----
        edge_src : np.ndarray
            源节点索引
        edge_dst : np.ndarray
            目标节点索引
        edge_attrs : np.ndarray
            边属性（偏移差值）
        """
        N = len(node_offsets)
        ct = valid_cube.shape[0] // 2
        ch = valid_cube.shape[1] // 2
        cw = valid_cube.shape[2] // 2

        # 计算所有节点对之间的距离
        distances = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = node_offsets[i] - node_offsets[j]
                    distances[i, j] = np.sqrt(np.sum(d ** 2))

        # KNN 边构建
        k = min(self.graph_cfg.edge_knn, N - 1)
        src_list, dst_list, attr_list = [], [], []

        for i in range(N):
            # 找到距离最近的 K 个邻居
            if i == 0:
                # 中心节点连接所有其他节点
                neighbors = list(range(1, N))
            else:
                dist_i = distances[i].copy()
                dist_i[i] = float('inf')
                neighbors = list(np.argsort(dist_i)[:k])

            for j in neighbors:
                src_list.append(i)
                dst_list.append(j)
                attr_list.append(node_offsets[i] - node_offsets[j])

        # 添加自环
        for i in range(N):
            src_list.append(i)
            dst_list.append(i)
            attr_list.append(np.zeros(3, dtype=np.float32))

        n = len(src_list)
        return (
            np.array(src_list, dtype=np.int64),
            np.array(dst_list, dtype=np.int64),
            np.array(attr_list, dtype=np.float32),
        )

    def _crop_cube(self, tc, hc, wc, radius):
        """裁剪时空子立方体"""
        t0 = max(0, tc - radius)
        t1 = min(self.T, tc + radius + 1)
        h0 = max(0, hc - radius)
        h1 = min(self.H, hc + radius + 1)
        w0 = max(0, wc - radius)
        w1 = min(self.W, wc + radius + 1)
        return (self.data[t0:t1, h0:h1, w0:w1].copy(),
                self.valid_mask[t0:t1, h0:h1, w0:w1].copy())
