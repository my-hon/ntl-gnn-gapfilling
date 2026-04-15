# 方案B：空间异质性感知的动态连边

## 概述

方案B是一种基于空间异质性的动态图构建方法，用于夜间灯光（NTL）时空数据的图神经网络预处理。核心思想是根据局部区域的空间异质性程度（NTL值的空间变异系数），动态调整图边的连接策略和属性权重。

## 灵感来源

- **Graph WaveNet** (arXiv:1906.00121) - 自适应邻接矩阵学习
- **DRTR** (arXiv:2406.17281) - 距离感知拓扑精炼

## 核心算法

### 1. 空间异质性指数

对每个中心位置，在局部时空立方体内计算变异系数：

```
H = std(NTL_values) / mean(NTL_values)
```

其中 `NTL_values` 是每个空间位置在时间维度上的均值。H 越大表示该区域 NTL 值的空间差异越大（如城市核心区），H 越小表示空间差异越小（如郊区）。

### 2. 动态时空权重

基于异质性指数，通过 sigmoid 函数计算动态权重：

```
w_spatial = sigmoid((H - H_threshold) * scale)
w_temporal = 1 - w_spatial
```

- **高异质性区域**（H > H_threshold）：w_spatial 大，侧重空间连接
- **低异质性区域**（H < H_threshold）：w_temporal 大，侧重时序连接

### 3. 动态边属性

边的属性使用动态权重混合空间和时序偏移：

```
edge_attr = w_spatial * spatial_offset + w_temporal * temporal_offset
```

### 4. 自适应连接策略

| 区域类型 | 异质性 | 空间策略 | 时序策略 |
|---------|--------|---------|---------|
| 城市核心 | 高 | 放松 Bresenham 截断，增加空间近邻 | 保持默认 |
| 郊区 | 低 | 保持默认 | 扩展时序连接窗口 |

## 模块结构

```
scheme_b_dynamic_edge_construction/
├── __init__.py                # 包入口，导出核心类
├── config_b.py                # 方案B配置（含异质性参数）
├── heterogeneity_analyzer.py  # 空间异质性分析器（Numba JIT）
├── dynamic_edge_builder.py    # 动态边构建器（Numba JIT）
├── graph_builder_b.py         # 方案B图构建器（与 v2 兼容）
└── README.md                  # 本文档
```

### 各模块说明

#### config_b.py

方案B的配置模块，在 v2 基础配置上增加 `HeterogeneityConfig`：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `h_threshold` | float | 0.25 | 异质性阈值 |
| `h_scale` | float | 10.0 | sigmoid 缩放因子 |
| `high_het_spatial_boost` | float | 1.5 | 高异质性区域路径放松系数 |
| `low_het_temporal_extend` | int | 3 | 低异质性区域时序扩展步数 |
| `heterogeneity_cube_radius` | int | 5 | 异质性计算感受野半径 |
| `min_valid_for_heterogeneity` | int | 8 | 最小有效像素数 |

#### heterogeneity_analyzer.py

空间异质性分析器，提供 Numba JIT 加速的异质性指数计算：

- `_compute_heterogeneity_single()` - 单立方体异质性计算（@njit）
- `_compute_spatial_temporal_weights()` - sigmoid 权重计算（@njit）
- `_compute_heterogeneity_batch()` - 批量异质性计算（@njit）
- `HeterogeneityAnalyzer` - 高级 Python 接口类

#### dynamic_edge_builder.py

动态边构建器，基于异质性权重调整 Bresenham 路径过滤：

- `_build_edges_dynamic_numba()` - 动态权重边构建（@njit）
- `_build_edges_dynamic_with_heterogeneity()` - 合并异质性+边构建（@njit）
- `DynamicEdgeBuilder` - 高级 Python 接口类

#### graph_builder_b.py

方案B图构建器，与 v2 的 `GraphBuilder` 接口兼容：

- `SubGraph` - 子图数据结构（扩展了异质性字段，基础接口不变）
- `GraphBuilderB` - 图构建器主类
- `build_single()` - 单位置构建
- `build_batch()` - 批量构建
- `build_batch_with_analysis()` - 批量构建 + 异质性分析报告

## 使用示例

### 基本用法

```python
import numpy as np
from adaptive_graph.scheme_b_dynamic_edge_construction import ConfigB, GraphBuilderB

# 创建配置
config = ConfigB()
config.heterogeneity.h_threshold = 0.25
config.heterogeneity.h_scale = 10.0

# 加载数据
data = np.load("ntl_data.npy").astype(np.float32)  # (T, H, W)
valid_mask = ~np.isnan(data)

# 创建构建器
builder = GraphBuilderB(config, data, valid_mask)

# 构建单个子图
subgraph = builder.build_single(tc=100, hc=200, wc=300)
if subgraph is not None:
    print(f"异质性指数: {subgraph.heterogeneity_index:.3f}")
    print(f"空间权重: {subgraph.w_spatial:.3f}")
    print(f"时序权重: {subgraph.w_temporal:.3f}")
    print(f"节点数: {subgraph.num_nodes}")
    print(f"边数: {len(subgraph.edge_index_src)}")

# 批量构建
positions = np.array([[100, 200, 300], [101, 200, 300]], dtype=np.int32)
graphs = builder.build_batch(positions)
```

### 带分析报告的批量构建

```python
graphs, analysis = builder.build_batch_with_analysis(positions)

print("异质性分布:")
print(f"  均值: {analysis['heterogeneity_distribution']['mean']:.3f}")
print(f"  标准差: {analysis['heterogeneity_distribution']['std']:.3f}")
print(f"  有效区域: {analysis['heterogeneity_distribution']['num_valid']}")

print("权重分布:")
print(f"  空间权重均值: {analysis['weight_distribution']['w_spatial_mean']:.3f}")
print(f"  时序权重均值: {analysis['weight_distribution']['w_temporal_mean']:.3f}")

print("边数统计:")
print(f"  平均边数: {analysis['edge_count_stats']['mean']:.1f}")
```

### 异质性分析器独立使用

```python
from adaptive_graph.scheme_b_dynamic_edge_construction import HeterogeneityAnalyzer

analyzer = HeterogeneityAnalyzer(
    cube_radius=5, h_threshold=0.25, h_scale=10.0, min_valid=8
)

# 分析单个立方体
H, ws, wt = analyzer.analyze_cube(data_cube, valid_cube)
print(f"H={H:.3f}, w_spatial={ws:.3f}, w_temporal={wt:.3f}")

# 批量分析
H_arr, ws_arr, wt_arr = analyzer.analyze_batch(full_data, full_valid, positions)
```

## 与 v2 的兼容性

方案B完全兼容 v2 的接口：

| 特性 | v2 | 方案B |
|------|----|----|
| SubGraph 基础字段 | center_pos, node_features, edge_index_src/dst, edge_attrs, center_value, num_nodes | 相同 + 扩展字段 |
| to_dict() | 返回标准字典 | 返回相同格式字典（忽略扩展字段） |
| Config | Config | ConfigB（继承 v2 配置） |
| 节点选择 | _select_nodes_numba | 复用 v2 |
| Bresenham LUT | BresenhamLUT | 复用 v2 |
| 缓存 | GraphCache | 复用 v2 |
| 边构建 | _build_edges_numba | _build_edges_dynamic_numba（方案B核心） |

## 性能说明

- 异质性计算和边构建均使用 Numba @njit 加速
- `_build_edges_dynamic_with_heterogeneity` 将异质性分析和边构建合并为单次 JIT 调用，避免 Python-Numba 数据传输开销
- 所有 Numba 函数启用 `cache=True`，首次编译后后续调用无需重新编译
- 提供 Numba 不可用时的纯 Python 回退
