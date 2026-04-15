# NTL-GNN Gapfilling

基于 GNN 的夜间灯光数据（NASA Black Marble VNP46A2）缺失值填补 —— 图构建与加速框架。

> 论文：*"Gap-filling method based on GNNs considering spatiotemporal anisotropic geometric relationship for VNP46A2 data"* (Xu et al., 2026, IJDE)

## 项目结构

```
ntl-gnn-gapfilling/
├── ntl_graph_accel/              # v1: GPU Bresenham 查找表加速
│   ├── config.py                 # 配置参数
│   ├── data_loader.py            # 数据加载与预处理
│   ├── gpu_bresenham.py          # 3D Bresenham 查找表 + GPU 加速
│   ├── graph_builder.py          # 核心图构建算法
│   ├── graph_cache.py            # LRU 缓存
│   ├── spatial_partitioner.py    # 空间瓦片并行
│   └── main.py                   # CLI 入口
│
├── ntl_graph_accel_v2/           # v2: Numba JIT 加速（推荐）
│   ├── config.py                 # 配置参数
│   ├── data_loader.py            # 数据加载与预处理
│   ├── jit_kernels.py            # Numba JIT 核心计算内核
│   ├── graph_builder.py          # 核心图构建算法
│   ├── spatial_partitioner.py    # 空间瓦片并行
│   └── main.py                   # CLI 入口
│
├── adaptive_graph/               # 自适应图结构探索方案
│   ├── 方案A_quality_adaptive_nodes/    # 质量驱动自适应节点数
│   ├── 方案B_dynamic_edge_construction/ # 空间异质性感知动态连边
│   └── 方案C_attention_graph_learning/  # 注意力图结构学习
│
├── deploy/                       # 一键部署脚本（Featurize 平台）
│   ├── 1_setup_env.sh            # 环境配置（conda + PyTorch + PyG）
│   ├── 2_prepare_data.sh         # TIF 数据转 NumPy
│   └── 3_run_benchmark.sh        # 基准测试
│
├── performance_report.docx       # v1 vs v2 性能分析报告
└── pyproject.toml                # 项目配置
```

## 图构建算法

### 核心流程

对每个目标像素（缺失值位置），构建一个时空子图用于 GNN 推理：

1. **子立方体提取**：以目标像素为中心，提取 `EXT_RANGE=6` 范围的时空立方体，不足时自适应扩展
2. **象限分配**：将立方体内所有体素按偏移方向分配到 6 个象限（Plan A：按最大偏移轴+符号）
3. **轮询节点选择**：按欧氏距离排序，以 1→2→3→4→5→6→1... 轮询方式从各象限选取 `search_node=32` 个邻居节点
4. **三类边构建**（节点选择时交错构建）：
   - **Type A - Bresenham 遮挡边**：沿 3D Bresenham 直线检测遮挡，连接到第一个可见节点
   - **Type B - 同时间/同空间辅助边**：共享时间或空间坐标的节点间建立连接（新节点需更近中心）
   - **Type C - 自环**：每个非中心节点的自环边

### 输出格式

```python
{
    "node_features": np.array (N+1, 1),   # 中心节点特征 = [-1.0]（待预测标记）
    "edge_index":    np.array (2, E),     # [src, dst]
    "edge_attr":     np.array (E, 3),     # 归一化 3D 偏移量（/8.0）
    "ground_truth":  np.array (N+1, 1),   # 所有节点的真实值
    "position":      np.array (1, 3),     # 中心位置 [t, h, w]
}
```

### 数据预处理

```
原始 TIF (uint16, 65535=NoData)
  → 65535 替换为 NaN
  → 质量标志 > 1 的像素替换为 NaN
  → 除以 10.0（还原真实辐射值，单位 nW cm⁻² sr⁻¹）
  → 除以 100.0（归一化，适配长尾分布）
```

### 训练采样

使用自然间断法（Natural Breaks）将有效像素分为 7 个类别，每类采样 20000 个：

| 类别 | 范围 | 说明 |
|------|------|------|
| 1 | [-inf, 0.001) | 极暗区域（背景/水体） |
| 2 | [0.001, 0.00325) | 微弱灯光 |
| 3 | [0.00325, 0.0065) | 低亮度区域 |
| 4 | [0.0065, 0.0125) | 中低亮度 |
| 5 | [0.0125, 0.025) | 中等亮度 |
| 6 | [0.025, 0.1) | 较亮区域 |
| 7 | [0.1, inf) | 高亮区域（城市核心） |

## 环境要求

| 依赖 | 版本 |
|------|------|
| Python | >= 3.10 |
| PyTorch | 2.4.x ~ 2.5.x (CUDA 12.1) |
| PyTorch Geometric | 与 PyTorch 版本匹配 |
| Numba | >= 0.58.0 |
| rasterio | >= 1.3.0 |
| numpy, scipy, tqdm | — |

## 快速开始

### 方式一：一键部署（Featurize 平台）

```bash
# 1. 克隆项目
git clone https://github.com/my-hon/ntl-gnn-gapfilling.git
cd ntl-gnn-gapfilling

# 2. 环境配置（conda + PyTorch + PyG + 依赖）
bash deploy/1_setup_env.sh

# 3. 准备数据（TIF → NumPy）
bash deploy/2_prepare_data.sh

# 4. 运行基准测试
bash deploy/3_run_benchmark.sh
```

### 方式二：手动安装

```bash
# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 PyG
TORCH_VER=$(python -c "import torch; print('.'.join(torch.__version__.split('+')[0].split('.')[:3]))")
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cu121.html"

# 安装其他依赖
pip install numpy scipy numba tqdm rasterio
```

## 使用方法

### 训练数据生成

```bash
python -m ntl_graph_accel_v2.main \
    --mode train \
    --input /path/to/beijing_2020_ntl.npy \
    --quality /path/to/beijing_2020_quality.npy \
    --output ./output/train \
    --sample-per-class 20000 \
    --search-node 32 \
    --ext-range 6
```

输出：`./output/train/graph_20000_32_{1..7}_forTrain.pkl`（每类别一个文件）

### 预测数据生成

```bash
python -m ntl_graph_accel_v2.main \
    --mode predict \
    --input /path/to/beijing_2020_ntl.npy \
    --quality /path/to/beijing_2020_quality.npy \
    --output ./output/predict \
    --search-node 32 \
    --ext-range 6
```

输出：`./output/predict/graph_*_forPred_{001..366}.pkl`（每天一个文件）

### 完整参数

```
--mode {train,predict}        运行模式
--input PATH                  NTL 数据路径 (.npy)
--quality PATH                质量标志路径 (.npy)
--output PATH                 输出目录
--sample-per-class N          每类采样数（默认 20000）
--search-node N               邻居节点数（默认 32）
--ext-range N                 子立方体扩展范围（默认 6）
--edge-scale FLOAT            边属性归一化因子（默认 8.0）
--edge-time N                 时间缓冲区（默认 50）
--edge-height N               高度缓冲区（默认 50）
--edge-width N                宽度缓冲区（默认 50）
--workers N                   并行进程数（默认 8）
--seed N                      随机种子（默认 0）
```

## 性能基准

测试环境：NVIDIA RTX A4000 (16GB), CUDA 12.1, Python 3.11.8

数据：北京 2020 年 Black Marble VNP46A2，形状 (366, 560, 666)，7 类 × 20000 = 140000 图

| 版本 | 总耗时 | 平均速度 | 加速比 |
|------|--------|---------|--------|
| v1 (GPU Bresenham LUT) | 1515.03s | 94 图/秒 | 1.0x |
| v2 (Numba JIT) | **1101.14s** | **129 图/秒** | **1.376x** |

### v2 性能提升原理

| 优化点 | v1 | v2 | 原理 |
|--------|----|----|------|
| Bresenham 计算 | LUT 查表（内存随机访问） | JIT 即时计算（顺序计算） | EXT_RANGE=6 路径短，计算开销低于哈希查找 |
| 象限分配 | Python 循环 | Numba JIT 编译 | ~8000 元素数组操作，JIT 接近 C 速度 |
| 距离计算 | Python 循环 | Numba JIT 编译 | 同上 |
| 磁盘 I/O | 加载 68920 条路径 (~5.4s) | 无 | 消除 LUT 加载开销 |
| 缓存开销 | 量化哈希查找 | 无 | 消除缓存键计算和查找 |

详细分析见 [performance_report.docx](performance_report.docx)。

## 数据说明

### 数据来源

NASA Black Marble VNP46A2 (Collection 2, Version 2)，经 HDF→GeoTIFF 转换后的逐日产品。

### 文件结构

```
Beijing/
├── DNB_BRDF-Corrected_NTL/          # BRDF 校正 NTL 辐射值（存在缺失）
│   └── YYYYDDD_50KM.tif             # 年积日命名
├── Gap_Filled_DNB_BRDF-Corrected_NTL/ # 官方填补版本（参考）
├── Mandatory_Quality_Flag/           # 强制质量标志（0=高质量, >1=低质量）
└── QF_Cloud_Mask/                    # 云掩膜标志
```

### 数据格式

| 属性 | 值 |
|------|-----|
| 格式 | GeoTIFF (单波段) |
| 数据类型 | uint16 |
| 空间分辨率 | ~500m (15 角秒) |
| 投影 | WGS84 (EPSG:4326) |
| NoData 值 | 65535 |
| 辐射值缩放 | 像素值 × 0.1 = nW cm⁻² sr⁻¹ |
| 质量标志 NoData | 255 |

## 自适应图结构探索

项目包含三种自适应图结构改进方案（实验性）：

| 方案 | 核心思想 | 论文参考 |
|------|---------|---------|
| 方案A | 质量驱动自适应节点数 | Beyond kNN (Adaptive Sparse Neighborhood) |
| 方案B | 空间异质性感知动态连边 | Graph WaveNet, DRTR |
| 方案C | 注意力机制学习图结构 | ASTGCN, Pro-GNN |

## 许可证

MIT
