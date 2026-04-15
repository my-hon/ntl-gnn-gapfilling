#!/bin/bash
# ================================================================
# 脚本1: 环境配置
# 在 Featurize 服务器上运行
# ================================================================
set -e

WORKDIR="/home/featurize/app/ntl-work"
cd "$WORKDIR"

echo "========================================"
echo "  步骤1: 创建 conda 环境 (Python 3.11)"
echo "========================================"

# 配置阿里云镜像
conda config --add channels https://mirrors.aliyun.com/conda/pkgs/main/
conda config --add channels https://mirrors.aliyun.com/conda/pkgs/r/
conda config --add channels https://mirrors.aliyun.com/conda/pkgs/msys2/
conda config --set show_channel_urls yes

# 创建环境（如果已存在则跳过）
if conda env list | grep -q "^ntl "; then
    echo "conda 环境 ntl 已存在，跳过创建"
else
    echo "创建 conda 环境 ntl (Python 3.11)..."
    conda create -n ntl python=3.11 -y
fi

echo ""
echo "========================================"
echo "  步骤2: 安装 uv"
echo "========================================"

eval "$(conda shell.bash hook)"
conda activate ntl

if command -v uv &>/dev/null; then
    echo "uv 已安装: $(uv --version)"
else
    echo "安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv 安装完成: $(uv --version)"
fi

echo ""
echo "========================================"
echo "  步骤3: 克隆仓库"
echo "========================================"

if [ -d "$WORKDIR/ntl-gnn-gapfilling" ]; then
    echo "仓库已存在，拉取最新代码..."
    cd "$WORKDIR/ntl-gnn-gapfilling"
    git pull
else
    echo "克隆仓库..."
    cd "$WORKDIR"
    git clone https://github.com/my-hon/ntl-gnn-gapfilling.git
fi

echo ""
echo "========================================"
echo "  步骤4: 安装 Python 依赖"
echo "========================================"

# 使用清华镜像安装 PyTorch (cu121)
echo "安装 PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# 获取 PyTorch 版本
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
TORCH_SERIES=$(python -c "import torch; v=torch.__version__.split('+')[0]; print('.'.join(v.split('.')[:2]))")
echo "PyTorch 版本: $TORCH_VER (系列: $TORCH_SERIES)"

# 安装 PyG
echo "安装 PyTorch Geometric..."
pip install torch_geometric \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "https://data.pyg.org/whl/torch-${TORCH_SERIES}+cu121.html" \
    --no-deps \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    || echo "PyG 扩展库部分安装失败（核心功能仍可用）"

# 安装其他依赖
echo "安装其他依赖..."
pip install numpy scipy numba tqdm rasterio \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

echo ""
echo "========================================"
echo "  步骤5: 验证安装"
echo "========================================"

python -c "
import torch
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA 可用:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本:     {torch.version.cuda}')
    print(f'GPU:           {torch.cuda.get_device_name(0)}')
    print(f'GPU 显存:      {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

try:
    import torch_geometric
    print(f'PyG:           {torch_geometric.__version__}')
except ImportError:
    print('PyG:           未安装')

try:
    import numba
    print(f'Numba:         {numba.__version__}')
except ImportError:
    print('Numba:         未安装')

try:
    import rasterio
    print(f'rasterio:      {rasterio.__version__}')
except ImportError:
    print('rasterio:      未安装')

import numpy as np
print(f'NumPy:         {np.__version__}')
"

echo ""
echo "========================================"
echo "  环境配置完成!"
echo "========================================"
echo ""
echo "后续步骤:"
echo "  1. 运行数据准备: bash $WORKDIR/ntl-gnn-gapfilling/deploy/2_prepare_data.sh"
echo "  2. 运行基准测试: bash $WORKDIR/ntl-gnn-gapfilling/deploy/3_run_benchmark.sh"
