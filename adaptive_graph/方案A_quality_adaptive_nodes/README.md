# 方案A：数据质量驱动的自适应节点数

## 方案名称

**Quality-Adaptive Node Selection for Spatiotemporal Graph Construction**

## 灵感来源

| 论文 | 核心思想 | 本方案借鉴点 |
|------|---------|-------------|
| Beyond kNN: Graph Neural Networks for Point Clouds with Adaptive Neighborhood (arXiv:2208.00604) | 使用最优传输理论为每个点自适应选择邻域大小 | 根据局部数据质量动态调整图节点数量 |
| Pro-GNN: Property Graph Neural Network (arXiv:2005.10203) | 通过稀疏约束学习最优图结构 | 低质量区域减少节点数，实现稀疏化 |

## 核心算法流程

### 1. 质量指标计算

对每个空间位置 (h, w)，在其时空邻域内计算三个质量指标：

```
rho (有效像素密度) = 邻域内有效像素数 / 邻域总面积
    - 使用积分图（Integral Image）加速矩形区域求和
    - 反映数据完整性

S (空间连续性) = exp(-10 * Var(邻域内有效像素均值))
    - 邻域内有效像素的空间分布方差越小，连续性越好
    - 反映数据空间结构的完整性

T (时序稳定性) = 1 / (1 + CV)
    - CV = std(values) / |mean(values)| 为变异系数
    - CV 越小，时序越稳定
    - 反映数据时间维度的一致性
```

### 2. 综合质量分数

```
Q = rho^alpha * S^beta * T^gamma
```

默认权重：alpha=0.5, beta=0.3, gamma=0.2（密度最重要，稳定性次要）

### 3. 自适应节点数

```
f(Q) = 0.5 + Q                    # Q in [0,1] -> f(Q) in [0.5, 1.5]
N = clip(N_base * f(Q), N_min, N_max)
```

| 质量分数 Q | 缩放因子 f(Q) | 节点数 N（N_base=36） |
|-----------|--------------|---------------------|
| 0.0（极差） | 0.5 | 18 |
| 0.5（中等） | 1.0 | 36 |
| 1.0（极好） | 1.5 | 54 |

### 4. 区域配额缩放

各区域（6个方向）的配额按自适应节点数等比例缩放：
```
quota_per_region = N_adaptive // num_regions
remainder = N_adaptive % num_regions
```

## 与基线（固定36节点）的区别

| 维度 | 基线（v2） | 方案A |
|------|-----------|-------|
| 节点数 | 固定 36 | 自适应 18~54 |
| 高质量区域 | 可能欠表示 | 更多节点，更精细 |
| 低质量区域 | 冗余节点 | 更少节点，减少噪声 |
| 计算开销 | 恒定 | 质量相关（整体可能更低） |
| 图表达能力 | 均匀 | 质量自适应 |
| 额外预处理 | 无 | 一次性质量图预计算 |

## 关键参数说明

### QualityAdaptiveConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `alpha` | float | 0.5 | 有效像素密度权重 |
| `beta` | float | 0.3 | 空间连续性权重 |
| `gamma` | float | 0.2 | 时序稳定性权重 |
| `n_min` | int | 18 | 最小节点数 |
| `n_max` | int | 54 | 最大节点数 |
| `scale_offset` | float | 0.5 | 缩放函数截距 |
| `scale_slope` | float | 1.0 | 缩放函数斜率 |
| `integral_window` | int | 5 | 积分图滑动窗口半径 |
| `temporal_window` | int | 3 | 时序窗口半径 |

### GraphConfigA

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `num_nodes_base` | int | 36 | 基准节点数 N_base |
| `initial_radius` | int | 4 | 初始搜索半径 |
| `max_radius` | int | 20 | 最大搜索半径 |
| `num_regions` | int | 6 | 空间区域数 |

## 使用方法

### 基本用法

```python
import numpy as np
from adaptive_graph.方案A_quality_adaptive_nodes import (
    ConfigA, GraphBuilderA
)

# 1. 准备数据
data = np.load("data.npy")           # (T, H, W) float32
valid_mask = ~np.isnan(data)         # (T, H, W) bool

# 2. 创建配置
config = ConfigA()
config.data.data_shape = data.shape
config.quality.alpha = 0.5           # 可调整权重
config.quality.n_min = 18
config.quality.n_max = 54

# 3. 创建构建器
builder = GraphBuilderA(config, data, valid_mask)

# 4. 预计算质量图（推荐，一次性开销）
builder.precompute_quality()

# 5. 构建子图
positions = np.array([[t, h, w] for t, h, w in valid_positions])
graphs = builder.build_batch(positions)

# 6. 查看统计
stats = builder.get_stats()
print(f"节点数分布: {stats['adaptive_counts']}")
```

### 与 v2 无缝切换

```python
# v2 用法
from ntl_graph_accel_v2 import Config, GraphBuilder
builder_v2 = GraphBuilder(config_v2, data, valid_mask)

# 方案A 用法（接口兼容）
from adaptive_graph.方案A_quality_adaptive_nodes import ConfigA, GraphBuilderA
builder_a = GraphBuilderA(config_a, data, valid_mask)

# 输出的 SubGraph 结构完全相同
graph = builder_a.build_single(tc, hc, wc)
# graph.num_nodes 可能是 18~54 中的任意值（而非固定 36）
```

### 自定义质量权重

```python
# 更重视空间连续性
config.quality.alpha = 0.2   # 密度权重降低
config.quality.beta = 0.6    # 连续性权重提高
config.quality.gamma = 0.2   # 稳定性不变

# 更激进的自适应范围
config.quality.n_min = 12    # 最低12个节点
config.quality.n_max = 72    # 最高72个节点
config.quality.scale_slope = 1.5  # 更陡峭的缩放曲线
```

## 预期效果

1. **计算效率提升**：低质量区域节点数减少（18 vs 36），减少约 50% 的冗余计算
2. **表示精度提升**：高质量区域节点数增加（54 vs 36），提升约 50% 的空间分辨率
3. **噪声抑制**：低质量区域的稀疏化自然起到去噪效果
4. **内存优化**：整体平均节点数可能低于 36（取决于数据质量分布），减少图数据内存占用
5. **兼容性**：输出的 SubGraph 结构不变，下游 GNN 模型无需修改即可使用变长图

## 文件结构

```
方案A_quality_adaptive_nodes/
  __init__.py           # 包入口，导出核心类
  config_a.py           # 方案A配置（继承v2，增加质量自适应参数）
  adaptive_selector.py  # 自适应节点数选择器（Numba JIT加速）
  graph_builder_a.py    # 方案A图构建器（继承v2逻辑，重写节点选择）
  README.md             # 本文档
```
