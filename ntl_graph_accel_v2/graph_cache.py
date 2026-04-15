"""
缓存复用模块（v2）
==================
与 v1 完全一致，直接复用。
"""

import os
import pickle
import logging
from collections import OrderedDict
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GraphTemplate:
    """图结构模板"""
    node_offsets: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_attrs: np.ndarray
    self_loop_indices: np.ndarray
    region_counts: np.ndarray
    radius: int

    def __reduce__(self):
        return (self.__class__, (
            self.node_offsets, self.edge_src, self.edge_dst,
            self.edge_attrs, self.self_loop_indices,
            self.region_counts, self.radius
        ))


class GraphCache:
    """图结构 LRU 缓存"""

    def __init__(self, cache_dir="./graph_cache", max_size=50000, quantization_step=2):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.quant_step = quantization_step
        self._cache: OrderedDict[str, GraphTemplate] = OrderedDict()
        self.hits = 0
        self.misses = 0
        os.makedirs(cache_dir, exist_ok=True)

    def quantize(self, tc, hc, wc):
        return (tc // self.quant_step, hc // self.quant_step, wc // self.quant_step)

    @staticmethod
    def _make_key(tc, hc, wc):
        return f"{tc}_{hc}_{wc}"

    def get(self, tc, hc, wc):
        qt, qh, qw = self.quantize(tc, hc, wc)
        key = self._make_key(qt, qh, qw)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        disk_path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(disk_path):
            try:
                with open(disk_path, 'rb') as f:
                    template = pickle.load(f)
                self._put(key, template)
                self.hits += 1
                return template
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
        self.misses += 1
        return None

    def put(self, tc, hc, wc, template, persist=True):
        qt, qh, qw = self.quantize(tc, hc, wc)
        key = self._make_key(qt, qh, qw)
        self._put(key, template)
        if persist:
            disk_path = os.path.join(self.cache_dir, f"{key}.pkl")
            try:
                with open(disk_path, 'wb') as f:
                    pickle.dump(template, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                logger.warning(f"保存缓存失败: {e}")

    def _put(self, key, template):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = template

    def get_hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self):
        return {
            "hits": self.hits, "misses": self.misses,
            "hit_rate": self.get_hit_rate(),
            "memory_entries": len(self._cache),
            "max_size": self.max_size,
            "quantization_step": self.quant_step
        }
