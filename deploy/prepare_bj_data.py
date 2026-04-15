"""
北京 Black Marble TIF 数据 -> NumPy 转换脚本
=============================================
将 /home/featurize/data/output_tif/Beijing/ 下的 TIF 文件
按年份转换为 (T, H, W) 的 NumPy 数组。

文件命名格式: YYYYDDD_50KM.tif (年积日)
- DNB_BRDF-Corrected_NTL: 原始 NTL 辐射值 (uint16, 65535=NoData, 缩放因子0.1)
- Mandatory_Quality_Flag: 质量标志 (uint8, 255=NoData)
- QF_Cloud_Mask: 云掩膜 (uint16, 65535=NoData)

注意:
  - 2020年第8天(2020008)等缺失天数自动填充为 NaN
  - 闰年366天，平年365天
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime, timedelta


def doy_to_date(year: int, doy: int) -> datetime:
    """年积日 -> 日期"""
    return datetime(year, 1, 1) + timedelta(days=doy - 1)


def date_to_doy(year: int, month: int, day: int) -> int:
    """日期 -> 年积日"""
    return (datetime(year, month, day) - datetime(year, 1, 1)).days + 1


def get_days_in_year(year: int) -> int:
    """获取某年的天数"""
    return 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365


def scan_tif_files(directory: str) -> dict:
    """
    扫描目录中的 TIF 文件，返回 {doy: filepath} 映射。
    文件名格式: YYYYDDD_50KM.tif
    """
    tif_map = {}
    if not os.path.exists(directory):
        print(f"  目录不存在: {directory}")
        return tif_map

    for fname in os.listdir(directory):
        if not fname.endswith('.tif'):
            continue
        # 解析文件名: YYYYDDD_50KM.tif
        try:
            name_part = fname.replace('_50KM.tif', '')
            year = int(name_part[:4])
            doy = int(name_part[4:])
            tif_map[doy] = os.path.join(directory, fname)
        except (ValueError, IndexError):
            continue

    return tif_map


def load_tif(filepath: str) -> np.ndarray:
    """加载单个 TIF 文件"""
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            return src.read(1)
    except ImportError:
        pass

    try:
        from osgeo import gdal
        ds = gdal.Open(filepath)
        if ds:
            band = ds.GetRasterBand(1)
            return band.ReadAsArray()
    except ImportError:
        pass

    try:
        from PIL import Image
        img = Image.open(filepath)
        return np.array(img)
    except ImportError:
        pass

    raise ImportError(
        "无法读取 TIF 文件。请安装 rasterio: pip install rasterio"
    )


def build_year_array(tif_map: dict, year: int, nodata_value=65535,
                     scale_factor=0.1, dtype=np.float32) -> np.ndarray:
    """
    从 TIF 文件构建完整的年数组 (T, H, W)。

    Parameters
    ----------
    tif_map : dict
        {doy: filepath} 映射
    year : int
        年份
    nodata_value : int
        NoData 像素值
    scale_factor : float
        缩放因子（实际值 = 像素值 * scale_factor）
    dtype : np.dtype
        输出数据类型

    Returns
    -------
    np.ndarray
        形状 (T, H, W)，缺失天填充为 NaN
    """
    num_days = get_days_in_year(year)

    # 先加载第一个有效文件确定空间尺寸
    first_doy = None
    first_data = None
    for doy in sorted(tif_map.keys()):
        try:
            first_data = load_tif(tif_map[doy])
            first_doy = doy
            break
        except Exception as e:
            print(f"  警告: 无法加载 DOY {doy}: {e}")

    if first_data is None:
        raise ValueError(f"无法加载任何 TIF 文件")

    H, W = first_data.shape
    print(f"  空间尺寸: {H} x {W}")
    print(f"  数据类型: {first_data.dtype}, 值范围: [{first_data.min()}, {first_data.max()}]")

    # 创建输出数组
    result = np.full((num_days, H, W), np.nan, dtype=dtype)

    # 填充数据
    loaded = 0
    missing_days = []
    for doy in range(1, num_days + 1):
        if doy in tif_map:
            try:
                raw = load_tif(tif_map[doy])
                # 将 NoData 替换为 NaN
                if nodata_value is not None:
                    mask = (raw == nodata_value)
                else:
                    mask = np.zeros_like(raw, dtype=bool)

                arr = raw.astype(dtype)
                if scale_factor != 1.0:
                    arr = arr * scale_factor
                arr[mask] = np.nan

                result[doy - 1] = arr
                loaded += 1
            except Exception as e:
                print(f"  警告: DOY {doy} 加载失败: {e}")
                missing_days.append(doy)
        else:
            missing_days.append(doy)

    print(f"  成功加载: {loaded}/{num_days} 天")
    if missing_days:
        print(f"  缺失天数 ({len(missing_days)}): {missing_days[:20]}{'...' if len(missing_days) > 20 else ''}")

    return result


def build_flag_array(tif_map: dict, year: int, nodata_value=255) -> np.ndarray:
    """
    构建质量标志数组 (T, H, W)，NoData 替换为 NaN。
    """
    num_days = get_days_in_year(year)

    first_doy = None
    first_data = None
    for doy in sorted(tif_map.keys()):
        try:
            first_data = load_tif(tif_map[doy])
            first_doy = doy
            break
        except Exception:
            continue

    if first_data is None:
        raise ValueError("无法加载任何 TIF 文件")

    H, W = first_data.shape
    result = np.full((num_days, H, W), np.nan, dtype=np.float32)

    loaded = 0
    for doy in range(1, num_days + 1):
        if doy in tif_map:
            try:
                raw = load_tif(tif_map[doy])
                arr = raw.astype(np.float32)
                if nodata_value is not None:
                    arr[raw == nodata_value] = np.nan
                result[doy - 1] = arr
                loaded += 1
            except Exception:
                pass

    print(f"  成功加载: {loaded}/{num_days} 天")
    return result


def main():
    parser = argparse.ArgumentParser(description="Black Marble TIF -> NumPy 转换")
    parser.add_argument('--input', type=str, required=True,
                        help='Beijing 数据根目录')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--year', type=int, default=2020,
                        help='年份 (默认 2020)')
    parser.add_argument('--scale', type=float, default=0.1,
                        help='NTL 缩放因子 (默认 0.1)')
    args = parser.parse_args()

    base_dir = args.input
    year = args.year
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    print(f"=" * 60)
    print(f"  Black Marble TIF -> NumPy 转换")
    print(f"  年份: {year}, 输出: {out_dir}")
    print(f"=" * 60)

    # ---- 1. DNB_BRDF-Corrected_NTL ----
    print(f"\n--- [1/3] DNB_BRDF-Corrected_NTL ---")
    ntl_dir = os.path.join(base_dir, "DNB_BRDF-Corrected_NTL")
    ntl_map = scan_tif_files(ntl_dir)
    print(f"  找到 {len(ntl_map)} 个 TIF 文件")

    ntl_data = build_year_array(ntl_map, year, nodata_value=65535,
                                 scale_factor=args.scale)
    ntl_path = os.path.join(out_dir, f"beijing_{year}_ntl.npy")
    np.save(ntl_path, ntl_data)
    print(f"  保存: {ntl_path} shape={ntl_data.shape}")

    # ---- 2. Mandatory_Quality_Flag ----
    print(f"\n--- [2/3] Mandatory_Quality_Flag ---")
    quality_dir = os.path.join(base_dir, "Mandatory_Quality_Flag")
    quality_map = scan_tif_files(quality_dir)
    print(f"  找到 {len(quality_map)} 个 TIF 文件")

    quality_data = build_flag_array(quality_map, year, nodata_value=255)
    quality_path = os.path.join(out_dir, f"beijing_{year}_quality.npy")
    np.save(quality_path, quality_data)
    print(f"  保存: {quality_path} shape={quality_data.shape}")

    # ---- 3. QF_Cloud_Mask ----
    print(f"\n--- [3/3] QF_Cloud_Mask ---")
    cloud_dir = os.path.join(base_dir, "QF_Cloud_Mask")
    cloud_map = scan_tif_files(cloud_dir)
    print(f"  找到 {len(cloud_map)} 个 TIF 文件")

    cloud_data = build_flag_array(cloud_map, year, nodata_value=65535)
    cloud_path = os.path.join(out_dir, f"beijing_{year}_cloud.npy")
    np.save(cloud_path, cloud_data)
    print(f"  保存: {cloud_path} shape={cloud_data.shape}")

    # ---- 统计 ----
    print(f"\n{'=' * 60}")
    print(f"  数据统计")
    print(f"{'=' * 60}")
    valid = ~np.isnan(ntl_data)
    total = ntl_data.size
    print(f"  NTL 数组: {ntl_data.shape}")
    print(f"  有效像素: {valid.sum()}/{total} ({100*valid.sum()/total:.1f}%)")
    print(f"  缺失像素: {(~valid).sum()}/{total} ({100*(~valid).sum()/total:.1f}%)")
    valid_values = ntl_data[valid]
    if len(valid_values) > 0:
        print(f"  有效值范围: [{valid_values.min():.2f}, {valid_values.max():.2f}]")
        print(f"  有效值均值: {valid_values.mean():.4f}")
        print(f"  有效值中位数: {np.median(valid_values):.4f}")

    print(f"\n  输出文件:")
    for f in [ntl_path, quality_path, cloud_path]:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"    {f} ({size_mb:.1f} MB)")

    print(f"\n  转换完成!")


if __name__ == '__main__':
    main()
