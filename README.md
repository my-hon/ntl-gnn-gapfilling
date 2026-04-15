# NTL Graph Accel

基于 GNN 的夜间灯光数据（Black Marble VNP46A2）缺失值填补 —— 子图生成加速框架。

## 背景

夜间灯光数据受云层遮挡存在大量时空缺失。本工具实现了论文 *"Gap-filling method based on GNNs considering spatiotemporal anisotropic geometric relationship for VNP46A2 data"* (Xu et al., 2026, IJDE) 中的图构建算法，并在此基础上引入三大加速策略，将子图生成流程从 **24 小时级降至分钟级**。

## 加速策略

| 策略 | 原理 | 预期加速 |
|------|------|----------|
| **空间分块并行** | 将研究区域划分为独立瓦片，多进程并行处理 | 6-7x |
| **Bresenham 查找表** | 预计算所有偏移的 3D Bresenham 路径，运行时直接查表 | 5-10x |
| **缓存复用** | 相似位置共享图结构模板，仅更新节点特征值 | 3-5x |
| **综合** | 三者叠加 | **~180x** |

## 项目结构

```
ntl_graph_accel/
├── __init__.py              # 包入口
├── __main__.py              # python -m 支持
├── config.py                # 全局配置参数（数据、图构建、加速）
├── data_loader.py           # 数据加载、有效/缺失位置识别、归一化
├── gpu_bresenham.py         # 3D Bresenham 查找表 + GPU 批量加速
├── graph_builder.py         # 核心图构建（自适应窗口、六区域节点选择、树状连边）
├── graph_cache.py           # LRU 缓存（空间哈希量化 + 磁盘持久化）
├── spatial_partitioner.py   # 空间瓦片划分 + 多进程并行调度
└── main.py                  # 命令行入口
```

## 环境要求

| 依赖 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.6.x (CUDA 12.4) |
| PyTorch Geometric | 与 PyTorch 版本匹配 |
| uv | >= 0.5.3 |

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd ntl-graph-accel

# 安装依赖（uv 自动从对应索引安装 CUDA 版本）
uv sync

# 如需开发/可视化工具
uv sync --extra dev --extra viz
```

## 使用方法

### 准备数据

将 NTL 数据保存为 NumPy 数组（`.npy` 格式），形状为 `(T, H, W)`，缺失值用 `NaN` 表示：

```python
import numpy as np

# data.npy: shape = (366, 476, 520)
# 前后各 10 天为时间缓冲区，空间四周 50 像素为空间缓冲区
data = np.load("beijing_2020.npy")  # NaN for missing/low-quality
```

### 构建缺失位置子图（填补模式）

为所有缺失像素位置构建子图，用于 GNN 推理填补：

```bash
python -m ntl_graph_accel --mode missing \
    --input beijing_2020.npy \
    --output ./output_graphs \
    --workers 8 \
    --tile-size 128 \
    --use-cache
```

输出 `./output_graphs/missing_graphs.pkl`，包含所有子图的列表。

### 构建训练数据子图（训练模式）

随机采样有效像素位置构建子图，自动划分 6:2:2 训练/验证/测试集：

```bash
python -m ntl_graph_accel --mode training \
    --input beijing_2020.npy \
    --output ./output_graphs \
    --workers 8
```

输出：
- `./output_graphs/train_graphs.pkl`
- `./output_graphs/val_graphs.pkl`
- `./output_graphs/test_graphs.pkl`

### 完整参数

```
usage: main.py [-h] --mode {missing,training} --input INPUT [--output OUTPUT]
               [--cache-dir CACHE_DIR] [--tile-size TILE_SIZE] [--workers WORKERS]
               [--use-cuda] [--use-cache | --no-cache] [--cache-quant CACHE_QUANT]
               [--num-nodes NUM_NODES] [--initial-radius INITIAL_RADIUS]
               [--max-radius MAX_RADIUS] [--buffer-size BUFFER_SIZE]
               [--temporal-buffer TEMPORAL_BUFFER]

参数说明:
  --mode              运行模式: missing（缺失填补）或 training（训练数据）
  --input             输入数据路径 (.npy)
  --output            输出目录 (默认: ./output_graphs)
  --cache-dir         缓存目录 (默认: ./graph_cache)
  --tile-size         空间瓦片大小，像素 (默认: 128)
  --workers           并行进程数 (默认: 8)
  --use-cuda          启用 CUDA 加速
  --use-cache / --no-cache  启用/禁用缓存复用 (默认: 启用)
  --cache-quant       缓存空间量化步长 (默认: 2)
  --num-nodes         图中节点数 (默认: 36)
  --initial-radius    时空立方体初始半窗口 (默认: 4)
  --max-radius        最大半窗口 (默认: 20)
  --buffer-size       空间缓冲区大小 (默认: 50)
  --temporal-buffer   时间缓冲区天数 (默认: 10)
```

## 子图数据结构

每个子图（`SubGraph`）包含以下字段，可直接用于 PyG 训练：

```python
@dataclass
class SubGraph:
    center_pos: np.ndarray       # 中心节点全局坐标 (t, h, w)
    node_features: np.ndarray    # 节点特征值 (N,)，已归一化
    edge_index_src: np.ndarray   # 边源索引 (E,)
    edge_index_dst: np.ndarray   # 边目标索引 (E,)
    edge_attrs: np.ndarray       # 边属性 (E, 3)，归一化后的 3D 偏移量
    center_value: float          # 中心节点真实值（ground truth）
    num_nodes: int               # 节点数
```

### 在 PyG 中使用

```python
import pickle
import torch
from torch_geometric.data import Data

with open("output_graphs/train_graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

# 转换为 PyG Data 对象
pyg_data_list = []
for g in graphs:
    data = Data(
        x=torch.from_numpy(g.node_features).float(),
        edge_index=torch.stack([
            torch.from_numpy(g.edge_index_src),
            torch.from_numpy(g.edge_index_dst)
        ], dim=0).long(),
        edge_attr=torch.from_numpy(g.edge_attrs).float(),
        y=torch.tensor([g.center_value]).float()
    )
    pyg_data_list.append(data)
```

## 性能参考

测试环境：CPU 单进程，模拟数据 (60×100×100)，查找表模式。

| 指标 | 数值 |
|------|------|
| 单图构建（首次） | ~0.5 ms |
| 单图构建（缓存命中） | ~0.04 ms |
| 缓存加速比 | ~12x |
| 批量构建速度 | ~984 图/秒 |
| 单图序列化大小 | ~1.6 KB |

## 许可证

MIT
