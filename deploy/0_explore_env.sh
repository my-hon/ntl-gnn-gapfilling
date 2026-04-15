#!/bin/bash
# ================================================================
# 环境探测脚本 - 在 Featurize 服务器上运行
# ================================================================
echo "========================================"
echo "  环境探测"
echo "========================================"

echo ""
echo "=== 系统信息 ==="
uname -a

echo ""
echo "=== GPU 信息 ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi 不可用"
echo "CUDA Version: $(nvidia-smi | grep 'CUDA Version' | awk '{print $4}')"

echo ""
echo "=== Python ==="
python --version 2>/dev/null
python3 --version 2>/dev/null
which python python3 2>/dev/null

echo ""
echo "=== Conda ==="
conda --version 2>/dev/null
conda info --envs 2>/dev/null

echo ""
echo "=== pip ==="
pip --version 2>/dev/null
pip3 --version 2>/dev/null

echo ""
echo "=== uv ==="
uv --version 2>/dev/null || echo "uv 未安装"

echo ""
echo "=== 磁盘空间 ==="
df -h /home/featurize

echo ""
echo "=== 工作目录 ==="
ls -la /home/featurize/app/ 2>/dev/null

echo ""
echo "=== 数据目录结构 ==="
echo "--- Beijing 目录 ---"
ls /home/featurize/data/output_tif/Beijing/ 2>/dev/null

echo ""
echo "--- DNB_BRDF-Corrected_NTL 文件数量和前5个 ---"
ls /home/featurize/data/output_tif/Beijing/DNB_BRDF-Corrected_NTL/ 2>/dev/null | head -5
echo "文件总数: $(ls /home/featurize/data/output_tif/Beijing/DNB_BRDF-Corrected_NTL/ 2>/dev/null | wc -l)"

echo ""
echo "--- Gap_Filled 文件数量和前5个 ---"
ls /home/featurize/data/output_tif/Beijing/Gap_Filled_DNB_BRDF\ Corrected_NTL/ 2>/dev/null | head -5
echo "文件总数: $(ls /home/featurize/data/output_tif/Beijing/Gap_Filled_DNB_BRDF\ Corrected_NTL/ 2>/dev/null | wc -l)"

echo ""
echo "--- Mandatory_Quality_Flag 文件数量和前5个 ---"
ls /home/featurize/data/output_tif/Beijing/Mandatory_Quality_Flag/ 2>/dev/null | head -5
echo "文件总数: $(ls /home/featurize/data/output_tif/Beijing/Mandatory_Quality_Flag/ 2>/dev/null | wc -l)"

echo ""
echo "--- QF_Cloud_Mask 文件数量和前5个 ---"
ls /home/featurize/data/output_tif/Beijing/QF_Cloud_Mask/ 2>/dev/null | head -5
echo "文件总数: $(ls /home/featurize/data/output_tif/Beijing/QF_Cloud_Mask/ 2>/dev/null | wc -l)"

echo ""
echo "=== TIF 文件详情（取一个样本）==="
SAMPLE=$(ls /home/featurize/data/output_tif/Beijing/DNB_BRDF-Corrected_NTL/*.tif 2>/dev/null | head -1)
if [ -n "$SAMPLE" ]; then
    echo "样本文件: $SAMPLE"
    echo "文件大小: $(ls -lh "$SAMPLE" | awk '{print $5}')"
    python3 -c "
import rasterio
with rasterio.open('$SAMPLE') as src:
    print(f'  波段数: {src.count}')
    print(f'  宽度: {src.width}')
    print(f'  高度: {src.height}')
    print(f'  数据类型: {src.dtypes[0]}')
    print(f'  CRS: {src.crs}')
    print(f'  Transform: {src.transform}')
    print(f'  NoData: {src.nodata}')
    import numpy as np
    data = src.read(1)
    print(f'  值范围: [{data.min()}, {data.max()}]')
    print(f'  NaN数量: {np.isnan(data.astype(float)).sum() if data.dtype != object else \"N/A\"}')
    print(f'  65535数量: {(data == 65535).sum()}')
" 2>/dev/null || echo "  rasterio 不可用，无法读取 TIF 详情"
else
    echo "未找到 TIF 文件"
fi

echo ""
echo "=== 网络测试 ==="
curl -s --connect-timeout 5 -o /dev/null -w "PyPI: %{http_code}\n" https://pypi.org/simple/ 2>/dev/null || echo "PyPI: 不可达"
curl -s --connect-timeout 5 -o /dev/null -w "PyTorch: %{http_code}\n" https://download.pytorch.org/whl/cu121/ 2>/dev/null || echo "PyTorch: 不可达"
curl -s --connect-timeout 5 -o /dev/null -w "阿里云镜像: %{http_code}\n" https://mirrors.aliyun.com/ 2>/dev/null || echo "阿里云镜像: 不可达"
curl -s --connect-timeout 5 -o /dev/null -w "清华镜像: %{http_code}\n" https://pypi.tuna.tsinghua.edu.cn/simple/ 2>/dev/null || echo "清华镜像: 不可达"

echo ""
echo "=== 完成 ==="
