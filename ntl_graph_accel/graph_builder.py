"""
核心图构建模块
==============
实现论文中的时空立方体图构建算法，集成GPU加速与缓存复用。
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict

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
    # 节点特征值 (N,) - 归一化后的NTL值
    node_features: np.ndarray
    # 边的源索引 (E,)
    edge_index_src: np.ndarray
    # 边的目标索引 (E,)
    edge_index_dst: np.ndarray
    # 边属性 (E, 3) - 归一化后的3D偏移量
    edge_attrs: np.ndarray
    # 中心节点真实值（ground truth）
    center_value: float
    # 图的节点数
    num_nodes: int

    def to_dict(self) -> Dict:
        """转为字典（便于序列化）"""
        return {
            'center_pos': self.center_pos,
            'node_features': self.node_features,
            'edge_index': np.stack([self.edge_index_src, self.edge_index_dst], axis=0),
            'edge_attrs': self.edge_attrs,
            'center_value': self.center_value,
            'num_nodes': self.num_nodes
        }


class GraphBuilder:
    """
    时空图构建器。
    
    实现论文Algorithm 1的加速版本：
    1. 自适应窗口扩展
    2. 六区域节点选择
    3. GPU批量Bresenham连边
    4. 缓存复用
    """

    def __init__(self, config: Config, data: np.ndarray, valid_mask: np.ndarray):
        """
        Parameters
        ----------
        config : Config
            全局配置
        data : np.ndarray
            完整NTL数据 (T, H, W)
        valid_mask : np.ndarray
            有效像素掩码 (T, H, W)
        """
        self.config = config
        self.data = data
        self.valid_mask = valid_mask
        self.T, self.H, self.W = data.shape

        # 初始化加速组件
        self.cache: Optional[GraphCache] = None
        self.lookup_table: Optional[BresenhamLookupTable] = None
        self.gpu_accel: Optional[GPUBresenhamAccelerator] = None

        self._init_accelerators()

    def _init_accelerators(self):
        """初始化加速器"""
        accel = self.config.accel

        # 缓存
        if accel.use_cache:
            self.cache = GraphCache(
                cache_dir=self.config.cache_dir,
                max_size=accel.cache_max_size,
                quantization_step=accel.cache_quantization
            )
            logger.info(f"缓存已启用, 量化步长={accel.cache_quantization}")

        # Bresenham查找表
        if accel.bresenham_lookup:
            self.lookup_table = BresenhamLookupTable(
                max_radius=self.config.graph.max_radius
            )
            # 尝试加载已有查找表
            lut_path = f"{self.config.cache_dir}/bresenham_lut.npz"
            import os
            if os.path.exists(lut_path):
                self.lookup_table.load(lut_path)
                logger.info("Bresenham查找表已从磁盘加载")
            else:
                self.lookup_table.build()
                self.lookup_table.save(lut_path)
                logger.info("Bresenham查找表已构建并保存")

        # GPU加速器
        if accel.use_cuda:
            try:
                self.gpu_accel = GPUBresenhamAccelerator(
                    max_radius=self.config.graph.max_radius,
                    max_path_len=self.config.graph.max_bresenham_len
                )
            except Exception as e:
                logger.warning(f"GPU加速器初始化失败: {e}, 将使用CPU模式")

    def build_single(self, tc: int, hc: int, wc: int) -> Optional[SubGraph]:
        """
        为单个位置构建子图。

        Parameters
        ----------
        tc, hc, wc : int
            中心坐标

        Returns
        -------
        SubGraph or None
            构建成功返回子图，数据不足返回None
        """
        graph_cfg = self.config.graph

        # 尝试缓存命中
        if self.cache is not None:
            template = self.cache.get(tc, hc, wc)
            if template is not None:
                return self._build_from_template(tc, hc, wc, template)

        # 自适应窗口扩展
        radius = graph_cfg.initial_radius
        cube, valid_cube, cube_origin = self._crop_cube(tc, hc, wc, radius)

        while True:
            # 检查各区域有效像素数
            region_valid_counts = self._count_region_valid(valid_cube, radius)
            min_required = self._get_region_quotas(graph_cfg.num_nodes)

            if all(region_valid_counts[i] >= min_required[i] for i in range(graph_cfg.num_regions)):
                break

            radius += 1
            if radius > graph_cfg.max_radius:
                logger.debug(f"位置({tc},{hc},{wc})窗口扩展超过最大值, 跳过")
                return None

            cube, valid_cube, cube_origin = self._crop_cube(tc, hc, wc, radius)

        # 选择节点
        node_info = self._select_nodes(cube, valid_cube, radius, graph_cfg.num_nodes)
        if node_info is None:
            return None

        node_offsets, node_features, region_ids = node_info

        # 构建边（使用GPU加速或查找表）
        edge_result = self._build_edges(
            node_offsets, valid_cube, radius, cube_origin
        )
        edge_src, edge_dst, edge_attrs, self_loops = edge_result

        # 归一化
        node_features = node_features / self.config.data.feature_scale
        edge_attrs = edge_attrs / self.config.data.edge_scale

        # 中心节点值
        center_value = self.data[tc, hc, wc]

        # 构建子图
        subgraph = SubGraph(
            center_pos=np.array([tc, hc, wc], dtype=np.int32),
            node_features=node_features.astype(np.float32),
            edge_index_src=edge_src.astype(np.int64),
            edge_index_dst=edge_dst.astype(np.int64),
            edge_attrs=edge_attrs.astype(np.float32),
            center_value=float(center_value),
            num_nodes=len(node_offsets)
        )

        # 存入缓存
        if self.cache is not None:
            template = GraphTemplate(
                node_offsets=node_offsets.copy(),
                edge_src=edge_src.copy(),
                edge_dst=edge_dst.copy(),
                edge_attrs=edge_attrs.copy(),
                self_loop_indices=self_loops.copy(),
                region_counts=np.array([np.sum(region_ids == i) for i in range(graph_cfg.num_regions)]),
                radius=radius
            )
            self.cache.put(tc, hc, wc, template)

        return subgraph

    def build_batch(self, positions: np.ndarray) -> List[SubGraph]:
        """
        批量构建子图。

        Parameters
        ----------
        positions : np.ndarray
            形状 (N, 3) 的坐标数组

        Returns
        -------
        List[SubGraph]
            成功构建的子图列表
        """
        graphs = []
        total = len(positions)

        for i, (tc, hc, wc) in enumerate(positions):
            tc, hc, wc = int(tc), int(hc), int(wc)
            graph = self.build_single(tc, hc, wc)
            if graph is not None:
                graphs.append(graph)

            if (i + 1) % 10000 == 0:
                logger.info(f"进度: {i+1}/{total}, "
                           f"已构建: {len(graphs)}, "
                           f"缓存命中率: {self.cache.get_hit_rate():.1%}" if self.cache else f"进度: {i+1}/{total}")

        return graphs

    def _crop_cube(
        self, tc: int, hc: int, wc: int, radius: int
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        """
        裁剪时空子立方体。

        Returns
        -------
        cube : np.ndarray
            NTL值立方体
        valid_cube : np.ndarray
            有效掩码立方体
        origin : (t0, h0, w0)
            立方体在全局数据中的起始坐标
        """
        t0 = max(0, tc - radius)
        t1 = min(self.T, tc + radius + 1)
        h0 = max(0, hc - radius)
        h1 = min(self.H, hc + radius + 1)
        w0 = max(0, wc - radius)
        w1 = min(self.W, wc + radius + 1)

        cube = self.data[t0:t1, h0:h1, w0:w1]
        valid_cube = self.valid_mask[t0:t1, h0:h1, w0:w1]

        return cube, valid_cube, (t0, h0, w0)

    def _count_region_valid(
        self, valid_cube: np.ndarray, radius: int
    ) -> np.ndarray:
        """
        计算六个区域中各区域的有效像素数。
        基于体对角线划分。
        """
        ct, ch, cw = valid_cube.shape[0] // 2, valid_cube.shape[1] // 2, valid_cube.shape[2] // 2

        # 六个区域的掩码（基于体对角线方向）
        regions = np.zeros(6, dtype=np.int32)

        # 获取所有有效像素的相对坐标
        valid_ts, valid_hs, valid_ws = np.where(valid_cube)

        if len(valid_ts) == 0:
            return regions

        # 相对于中心的偏移
        dt = valid_ts - ct
        dh = valid_hs - ch
        dw = valid_ws - cw

        # 基于体对角线划分区域
        # 区域编号基于(dt, dh, dw)的符号组合
        for i in range(len(dt)):
            d = (dt[i], dh[i], dw[i])
            region_id = self._get_region_id(d)
            if 0 <= region_id < 6:
                regions[region_id] += 1

        return regions

    @staticmethod
    def _get_region_id(d: Tuple[int, int, int]) -> int:
        """
        根据偏移方向确定区域编号。
        使用体对角线将3D空间划分为6个区域。
        """
        dt, dh, dw = d

        # 归一化方向向量
        length = (dt**2 + dh**2 + dw**2) ** 0.5
        if length == 0:
            return 0

        # 基于主方向和次方向确定区域
        abs_d = sorted([(abs(dt), 0), (abs(dh), 1), (abs(dw), 2)], reverse=True)

        # 主方向决定大区域（0-1: 时间, 2-3: 纬度, 4-5: 经度）
        primary = abs_d[0][1]
        sign = [dt, dh, dw][primary]

        if primary == 0:  # 时间主导
            return 0 if sign >= 0 else 1
        elif primary == 1:  # 纬度主导
            return 2 if sign >= 0 else 3
        else:  # 经度主导
            return 4 if sign >= 0 else 5

    def _get_region_quotas(self, num_nodes: int) -> List[int]:
        """计算各区域的最小节点配额"""
        q, rem = divmod(num_nodes, 6)
        quotas = [q + 1 if i < rem else q for i in range(6)]
        return quotas

    def _select_nodes(
        self,
        cube: np.ndarray,
        valid_cube: np.ndarray,
        radius: int,
        num_nodes: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        从立方体中选择图节点。

        Returns
        -------
        node_offsets : (N, 3) 节点在立方体中的相对坐标
        node_features : (N,) 节点NTL值
        region_ids : (N,) 各节点所属区域编号
        """
        ct = cube.shape[0] // 2
        ch = cube.shape[1] // 2
        cw = cube.shape[2] // 2

        # 获取所有有效像素
        valid_ts, valid_hs, valid_ws = np.where(valid_cube)
        if len(valid_ts) == 0:
            return None

        # 计算到中心的距离
        dt = valid_ts - ct
        dh = valid_hs - ch
        dw = valid_ws - cw
        distances = np.sqrt(dt**2 + dh**2 + dw**2)

        # 按距离排序
        sort_idx = np.argsort(distances)
        valid_ts = valid_ts[sort_idx]
        valid_hs = valid_hs[sort_idx]
        valid_ws = valid_ws[sort_idx]
        distances = distances[sort_idx]

        # 计算各区域配额
        quotas = self._get_region_quotas(num_nodes)
        region_counts = [0] * 6

        # 循环分配节点到各区域
        selected_offsets = []
        selected_features = []
        selected_regions = []

        flag = 0  # 当前需要的区域编号
        for i in range(len(valid_ts)):
            if len(selected_offsets) >= num_nodes:
                break

            d = (int(valid_ts[i] - ct), int(valid_hs[i] - ch), int(valid_ws[i] - cw))
            rid = self._get_region_id(d)

            if region_counts[rid] < quotas[rid]:
                selected_offsets.append(d)
                selected_features.append(cube[valid_ts[i], valid_hs[i], valid_ws[i]])
                selected_regions.append(rid)
                region_counts[rid] += 1

        if len(selected_offsets) < num_nodes:
            # 配额未满，放宽条件
            for i in range(len(valid_ts)):
                if len(selected_offsets) >= num_nodes:
                    break
                d = (int(valid_ts[i] - ct), int(valid_hs[i] - ch), int(valid_ws[i] - cw))
                if d not in selected_offsets:
                    selected_offsets.append(d)
                    selected_features.append(cube[valid_ts[i], valid_hs[i], valid_ws[i]])
                    selected_regions.append(self._get_region_id(d))

        if len(selected_offsets) < 2:
            return None

        return (
            np.array(selected_offsets, dtype=np.int32),
            np.array(selected_features, dtype=np.float32),
            np.array(selected_regions, dtype=np.int32)
        )

    def _build_edges(
        self,
        node_offsets: np.ndarray,
        valid_cube: np.ndarray,
        radius: int,
        cube_origin: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        构建图的边连接。
        使用查找表或GPU加速Bresenham算法。

        Returns
        -------
        edge_src : (E,) 源节点索引
        edge_dst : (E,) 目标节点索引
        edge_attrs : (E, 3) 边属性（3D偏移量）
        self_loops : (N,) 自环节点索引
        """
        N = len(node_offsets)
        center_offset = (0, 0, 0)  # 中心节点在立方体中始终是(0,0,0)

        # 构建偏移量到节点索引的映射
        offset_to_idx = {}
        for i, off in enumerate(node_offsets):
            offset_to_idx[tuple(off)] = i

        edge_src_list = []
        edge_dst_list = []
        edge_attrs_list = []
        self_loops = []

        if self.lookup_table is not None:
            # 使用查找表
            for i in range(N):
                off = node_offsets[i]
                if np.array_equal(off, center_offset):
                    continue  # 跳过中心节点

                dt, dh, dw = int(off[0]), int(off[1]), int(off[2])
                path = self.lookup_table.lookup(dt, dh, dw)

                # 过滤NaN中间点
                ct = valid_cube.shape[0] // 2
                ch = valid_cube.shape[1] // 2
                cw = valid_cube.shape[2] // 2

                first_valid = None
                for p in path:
                    pt, ph, pw = int(p[0]) + ct, int(p[1]) + ch, int(p[2]) + cw
                    if 0 <= pt < valid_cube.shape[0] and \
                       0 <= ph < valid_cube.shape[1] and \
                       0 <= pw < valid_cube.shape[2]:
                        if valid_cube[pt, ph, pw]:
                            key = (int(p[0]), int(p[1]), int(p[2]))
                            if key in offset_to_idx:
                                first_valid = offset_to_idx[key]
                                break
                    # 遇到NaN则截断
                    break

                if first_valid is not None:
                    edge_src_list.append(i)
                    edge_dst_list.append(first_valid)
                    # 边属性：当前节点相对于目标节点的偏移
                    target_off = node_offsets[first_valid]
                    edge_attrs_list.append(off - target_off)
                else:
                    # 无遮挡，直接连向中心节点
                    edge_src_list.append(i)
                    edge_dst_list.append(0)  # 中心节点索引
                    edge_attrs_list.append(off)

                # 自环
                self_loops.append(i)

        else:
            # CPU回退模式
            for i in range(N):
                off = node_offsets[i]
                if np.array_equal(off, center_offset):
                    continue

                # 简化Bresenham
                first_valid = self._bresenham_find_first(
                    off, center_offset, offset_to_idx, valid_cube
                )

                if first_valid is not None:
                    edge_src_list.append(i)
                    edge_dst_list.append(first_valid)
                    target_off = node_offsets[first_valid]
                    edge_attrs_list.append(off - target_off)
                else:
                    edge_src_list.append(i)
                    edge_dst_list.append(0)
                    edge_attrs_list.append(off)

                self_loops.append(i)

        if len(edge_src_list) == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.zeros((0, 3), dtype=np.float32),
                np.array([], dtype=np.int64)
            )

        return (
            np.array(edge_src_list, dtype=np.int64),
            np.array(edge_dst_list, dtype=np.int64),
            np.array(edge_attrs_list, dtype=np.float32),
            np.array(self_loops, dtype=np.int64)
        )

    @staticmethod
    def _bresenham_find_first(
        start: np.ndarray,
        end: np.ndarray,
        offset_to_idx: dict,
        valid_cube: np.ndarray
    ) -> Optional[int]:
        """CPU版Bresenham查找第一个有效中间节点"""
        dt, dh, dw = int(end[0] - start[0]), int(end[1] - start[1]), int(end[2] - start[2])
        steps = max(abs(dt), abs(dh), abs(dw))

        if steps == 0:
            return None

        ct = valid_cube.shape[0] // 2
        ch = valid_cube.shape[1] // 2
        cw = valid_cube.shape[2] // 2

        x, y, z = int(start[0]), int(start[1]), int(start[2])
        dx = (1 if dt > 0 else -1) if dt != 0 else 0
        dy = (1 if dh > 0 else -1) if dh != 0 else 0
        dz = (1 if dw > 0 else -1) if dw != 0 else 0
        sdx, sdy, sdz = abs(dt), abs(dh), abs(dw)
        if sdx == 0: sdx = 1
        if sdy == 0: sdy = 1
        if sdz == 0: sdz = 1

        ey, ez = -(steps // 2), -(steps // 2)

        for _ in range(steps):
            y_cur, z_cur = y, z
            ey += sdy
            ez += sdz
            if ey >= 0:
                y_cur += dy
                ey -= steps
            if ez >= 0:
                z_cur += dz
                ez -= steps
            x += dx

            key = (x, y_cur, z_cur)
            # 检查是否为有效节点
            pt, ph, pw = x + ct, y_cur + ch, z_cur + cw
            if 0 <= pt < valid_cube.shape[0] and \
               0 <= ph < valid_cube.shape[1] and \
               0 <= pw < valid_cube.shape[2]:
                if not valid_cube[pt, ph, pw]:
                    break  # 遇到NaN截断
                if key in offset_to_idx:
                    return offset_to_idx[key]
            else:
                break

        return None

    def _build_from_template(
        self, tc: int, hc: int, wc: int, template: GraphTemplate
    ) -> SubGraph:
        """从缓存模板快速构建子图（仅更新节点特征值）"""
        radius = template.radius
        cube, valid_cube, _ = self._crop_cube(tc, hc, wc, radius)

        ct = cube.shape[0] // 2
        ch = cube.shape[1] // 2
        cw = cube.shape[2] // 2

        # 从立方体中读取节点特征值
        node_features = np.array([
            cube[off[0] + ct, off[1] + ch, off[2] + cw]
            for off in template.node_offsets
        ], dtype=np.float32)

        node_features = node_features / self.config.data.feature_scale

        center_value = self.data[tc, hc, wc]

        return SubGraph(
            center_pos=np.array([tc, hc, wc], dtype=np.int32),
            node_features=node_features,
            edge_index_src=template.edge_src.copy(),
            edge_index_dst=template.edge_dst.copy(),
            edge_attrs=template.edge_attrs / self.config.data.edge_scale,
            center_value=float(center_value),
            num_nodes=len(template.node_offsets)
        )
