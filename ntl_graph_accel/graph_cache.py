"""
缓存复用模块
============
基于空间哈希的LRU缓存机制。
相似位置的子图结构可复用，仅更新节点特征值。
"""

import os
import pickle
import hashlib
import logging
from collections import OrderedDict
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GraphTemplate:
    """
    图结构模板。
    存储可复用的图拓扑结构（不含节点特征值）。
    """
    # 节点在立方体中的相对坐标 (N, 3)
    node_offsets: np.ndarray
    # 边的源节点索引 (E,)
    edge_src: np.ndarray
    # 边的目标节点索引 (E,)
    edge_dst: np.ndarray
    # 边属性（相对3D偏移量） (E, 3)
    edge_attrs: np.ndarray
    # 自环边索引
    self_loop_indices: np.ndarray
    # 各区域分配的节点数
    region_counts: np.ndarray
    # 使用的半径
    radius: int

    def __reduce__(self):
        """支持pickle序列化"""
        return (self.__class__, (
            self.node_offsets, self.edge_src, self.edge_dst,
            self.edge_attrs, self.self_loop_indices,
            self.region_counts, self.radius
        ))


class GraphCache:
    """
    图结构LRU缓存。
    
    核心思想：
    1. 将位置量化到网格，相近位置共享图结构模板
    2. 缓存命中时仅更新节点特征值，跳过耗时的Bresenham计算
    3. 支持持久化存储，跨运行复用
    """

    def __init__(
        self,
        cache_dir: str = "./graph_cache",
        max_size: int = 50000,
        quantization_step: int = 2
    ):
        """
        Parameters
        ----------
        cache_dir : str
            缓存文件存储目录
        max_size : int
            内存中LRU缓存最大条目数
        quantization_step : int
            空间量化步长（像素），步长越大缓存命中率越高但精度越低
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.quant_step = quantization_step

        # 内存LRU缓存
        self._cache: OrderedDict[str, GraphTemplate] = OrderedDict()

        # 统计
        self.hits = 0
        self.misses = 0

        os.makedirs(cache_dir, exist_ok=True)

    def quantize(self, tc: int, hc: int, wc: int) -> Tuple[int, int, int]:
        """量化位置坐标"""
        return (
            tc // self.quant_step,
            hc // self.quant_step,
            wc // self.quant_step
        )

    @staticmethod
    def _make_key(tc: int, hc: int, wc: int) -> str:
        """生成缓存键"""
        return f"{tc}_{hc}_{wc}"

    def get(self, tc: int, hc: int, wc: int) -> Optional[GraphTemplate]:
        """
        查询缓存。

        Returns
        -------
        GraphTemplate or None
            命中返回模板，未命中返回None
        """
        qt, qh, qw = self.quantize(tc, hc, wc)
        key = self._make_key(qt, qh, qw)

        if key in self._cache:
            # LRU: 移到末尾
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]

        # 尝试从磁盘加载
        disk_path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(disk_path):
            try:
                with open(disk_path, 'rb') as f:
                    template = pickle.load(f)
                # 放入内存缓存
                self._put(key, template)
                self.hits += 1
                return template
            except Exception as e:
                logger.warning(f"加载缓存文件失败: {disk_path}, 错误: {e}")

        self.misses += 1
        return None

    def put(
        self,
        tc: int, hc: int, wc: int,
        template: GraphTemplate,
        persist: bool = True
    ):
        """
        存入缓存。

        Parameters
        ----------
        tc, hc, wc : int
            中心坐标
        template : GraphTemplate
            图结构模板
        persist : bool
            是否持久化到磁盘
        """
        qt, qh, qw = self.quantize(tc, hc, wc)
        key = self._make_key(qt, qh, qw)
        self._put(key, template)

        if persist:
            disk_path = os.path.join(self.cache_dir, f"{key}.pkl")
            try:
                with open(disk_path, 'wb') as f:
                    pickle.dump(template, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                logger.warning(f"保存缓存文件失败: {disk_path}, 错误: {e}")

    def _put(self, key: str, template: GraphTemplate):
        """放入内存LRU缓存"""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                # 淘汰最旧的
                self._cache.popitem(last=False)
            self._cache[key] = template

    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.get_hit_rate(),
            "memory_entries": len(self._cache),
            "max_size": self.max_size,
            "quantization_step": self.quant_step
        }

    def clear_memory(self):
        """清空内存缓存"""
        self._cache.clear()
        logger.info("内存缓存已清空")

    def clear_disk(self):
        """清空磁盘缓存"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("磁盘缓存已清空")

    def preload_from_disk(self, max_files: int = 100000):
        """预加载磁盘缓存到内存"""
        cache_files = list(Path(self.cache_dir).glob("*.pkl"))
        cache_files = cache_files[:max_files]

        loaded = 0
        for f in cache_files:
            try:
                with open(f, 'rb') as fp:
                    template = pickle.load(fp)
                key = f.stem
                self._put(key, template)
                loaded += 1
            except Exception:
                pass

        logger.info(f"预加载缓存: {loaded}/{len(cache_files)} 条目")
