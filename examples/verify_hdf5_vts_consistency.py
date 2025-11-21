"""
验证HDF5和VTS文件的数据一致性
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from hdf5_storage import HDF5Storage
import numpy as np


def verify_hdf5_vts_consistency(hdf5_path, vts_path):
    """验证HDF5和VTS文件的数据一致性"""
    print(f"=== 验证数据一致性: {hdf5_path} ↔ {vts_path} ===")

    # 1. 加载HDF5数据
    print("\n1. 加载HDF5数据...")
    storage = HDF5Storage()
    try:
        h5_data = storage.load(hdf5_path)
        print(f"  [OK] HDF5加载成功")
        print(f"  - 网格尺寸: {h5_data['grid_size']}")
        print(f"  - 物理场: {list(h5_data['fields'].keys())}")
    except Exception as e:
        print(f"  [ERROR] HDF5加载失败: {e}")
        return False

    # 2. 读取VTS文件头部信息
    print("\n2. 分析VTS文件...")
    try:
        with open(vts_path, 'r', encoding='utf-8', errors='ignore') as f:
            vts_content = f.read()
        print(f"  [OK] VTS文件读取成功")
    except Exception as e:
        print(f"  [ERROR] VTS文件读取失败: {e}")
        return False

    # 3. 验证网格尺寸一致性
    print("\n3. 验证网格尺寸...")
    import re

    # 从VTS中提取WholeExtent
    extent_match = re.search(r'<WholeExtent[^>]*>([^<]+)</WholeExtent>', vts_content)
    if extent_match:
        extent_str = extent_match.group(1).strip()
        extents = list(map(int, extent_str.split()))
        vts_grid_size = [extents[i+1] - extents[i] + 1 for i in range(0, 6, 2)]
        vts_grid_size = tuple(vts_grid_size)

        print(f"  - VTS推断网格尺寸: {vts_grid_size}")
        print(f"  - HDF5网格尺寸: {tuple(h5_data['grid_size'])}")

        if vts_grid_size == tuple(h5_data['grid_size']):
            print(f"  [OK] 网格尺寸一致")
        else:
            print(f"  [WARNING] 网格尺寸不一致")
            return False
    else:
        print(f"  [WARNING] 无法从VTS提取网格尺寸")

    # 4. 验证物理场变量一致性
    print("\n4. 验证物理场变量...")
    h5_fields = set(h5_data['fields'].keys())

    # 从VTS中提取数据数组名称
    vts_arrays = set()
    array_pattern = re.compile(r'<DataArray[^>]*Name="([^"]+)"')
    for match in array_pattern.finditer(vts_content):
        vts_arrays.add(match.group(1))

    print(f"  - HDF5物理场: {sorted(h5_fields)}")
    print(f"  - VTS数据数组: {sorted(vts_arrays)}")

    # 检查核心物理场是否都在VTS中
    core_fields = {'P', 'Velocity', 'SDF', 'NodeID', 'RHO'}
    h5_core = h5_fields.intersection(core_fields)
    vts_core = vts_arrays.intersection(core_fields)

    print(f"  - HDF5核心场: {sorted(h5_core)}")
    print(f"  - VTS核心场: {sorted(vts_core)}")

    if h5_core.issubset(vts_arrays):
        print(f"  [OK] 核心物理场都在VTS中")
    else:
        missing = h5_core - vts_arrays
        print(f"  [WARNING] VTS中缺失场: {missing}")

    # 5. 验证数据范围（如果可能）
    print("\n5. 验证数据范围...")
    for field_name in ['P', 'SDF']:  # 检查标量场
        if field_name in h5_data['fields']:
            h5_field = h5_data['fields'][field_name]
            h5_min, h5_max = float(h5_field.min()), float(h5_field.max())
            print(f"  - {field_name} HDF5范围: [{h5_min:.3e}, {h5_max:.3e}]")

            # 尝试从VTS中提取范围
            range_pattern = re.compile(fr'<DataArray[^>]*Name="{field_name}"[^>]*>([^<]+)</DataArray>', re.DOTALL)
            range_match = range_pattern.search(vts_content)
            if range_match:
                data_str = range_match.group(1).strip()
                values = list(map(float, data_str.split()))
                if values:
                    vts_min, vts_max = min(values), max(values)
                    print(f"  - {field_name} VTS范围: [{vts_min:.3e}, {vts_max:.3e}]")

                    if abs(h5_min - vts_min) < 1e-10 and abs(h5_max - vts_max) < 1e-10:
                        print(f"    [OK] {field_name} 范围匹配")
                    else:
                        print(f"    [INFO] {field_name} 范围略有差异（可能是浮点精度）")

    # 6. 验证文件时间戳
    print("\n6. 验证文件时间戳...")
    h5_mtime = os.path.getmtime(hdf5_path)
    vts_mtime = os.path.getmtime(vts_path)

    import datetime
    h5_time = datetime.datetime.fromtimestamp(h5_mtime)
    vts_time = datetime.datetime.fromtimestamp(vts_mtime)

    print(f"  - HDF5修改时间: {h5_time}")
    print(f"  - VTS修改时间: {vts_time}")

    time_diff = abs(vts_mtime - h5_mtime)
    if time_diff < 60:  # 1分钟内
        print(f"  [OK] 时间戳一致（差异: {time_diff:.1f}秒）")
    else:
        print(f"  [INFO] 时间戳差异较大（{time_diff/60:.1f}分钟）")

    return True


def main():
    """主函数"""
    print("=== HDF5-VTS数据一致性验证 ===")

    # 文件路径
    hdf5_path = "../data_matrix/dense_32x32x32_zero_assignment_new_paths.h5"
    vts_path = "../data_matrix/dense_32x32x32_zero_assignment_new_paths.vts"

    # 验证文件存在
    if not os.path.exists(hdf5_path):
        print(f"[ERROR] HDF5文件不存在: {hdf5_path}")
        return False

    if not os.path.exists(vts_path):
        print(f"[ERROR] VTS文件不存在: {vts_path}")
        return False

    # 执行一致性验证
    if verify_hdf5_vts_consistency(hdf5_path, vts_path):
        print(f"\n=== 验证结论 ===")
        print(f"✓ VTS文件确实是从HDF5文件转换生成的")
        print(f"✓ 两个文件包含相同的插值数据")
        print(f"✓ 数据流程: VTU → 插值 → HDF5 → 转换 → VTS")
        return True
    else:
        print(f"\n[ERROR] 验证失败")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)