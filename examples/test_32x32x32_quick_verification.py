"""
快速验证32x32x32网格插值，使用新的data_UNS目录结构
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
from interpolator_optimized import OptimizedGridInterpolator
from hdf5_storage import HDF5Storage
from datetime import datetime


def main():
    """主测试函数"""
    print("=== 32x32x32网格插值快速验证（新目录结构） ===")

    # 初始化组件
    reader = VTKReader()
    interpolator = OptimizedGridInterpolator(
        grid_size=(32, 32, 32),  # 使用较小的网格快速验证
        method='linear',
        use_sdf=True,
        batch_size=10000
    )
    storage = HDF5Storage()

    # 输入文件路径（新目录结构）
    vtu_file = "../data_UNS/vessel/170/Part.0.Zone.1.vtu"
    print(f"输入文件: {vtu_file}")

    # 验证文件存在
    if not os.path.exists(vtu_file):
        print(f"[ERROR] 文件不存在: {vtu_file}")
        return False

    # 输出文件路径
    output_h5 = "../data_matrix/dense_32x32x32_zero_assignment_new_paths.h5"
    output_vts = "../data_matrix/dense_32x32x32_zero_assignment_new_paths.vts"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)

    try:
        # 1. 读取VTU数据
        print(f"\n1. 读取VTU数据...")
        vtu_data = reader.read_vtu(vtu_file)
        print(f"  [OK] 成功读取VTU文件")
        print(f"  - 网格类型: {vtu_data['type']}")
        print(f"  - 点数: {vtu_data['num_points']:,}")
        print(f"  - 单元数: {vtu_data['num_cells']:,}")

        # 显示可用的物理场变量
        if 'point_data' in vtu_data:
            fields = list(vtu_data['point_data'].keys())
            print(f"  - 可用物理场: {fields}")
        else:
            print(f"  - [WARNING] 未找到点数据")
            return False

        # 2. 执行插值
        print(f"\n2. 执行32x32x32网格插值...")
        start_time = datetime.now()

        interpolated_data = interpolator.interpolate(vtu_data, fields=fields)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"  [OK] 插值完成，耗时: {duration:.1f}秒")
        print(f"  - 网格尺寸: {interpolated_data['grid_size']}")
        print(f"  - SDF使用: {interpolated_data['sdf_used']}")

        # 显示插值结果统计
        for field_name, field_data in interpolated_data['fields'].items():
            import numpy as np
            field_info = {
                'shape': field_data.shape,
                'min': float(field_data.min()),
                'max': float(field_data.max()),
                'mean': float(field_data.mean()),
                'nan_count': int(np.sum(np.isnan(field_data))),
                'zero_count': int(np.sum(field_data == 0))
            }
            print(f"  - {field_name}:")
            print(f"    形状: {field_info['shape']}")
            print(f"    范围: [{field_info['min']:.3e}, {field_info['max']:.3e}]")
            print(f"    均值: {field_info['mean']:.3e}")
            print(f"    NaN点数: {field_info['nan_count']}")
            print(f"    零值点数: {field_info['zero_count']}")

        # 3. 保存HDF5文件
        print(f"\n3. 保存HDF5文件...")
        metadata = {
            'source_file': vtu_file,
            'grid_size': interpolated_data['grid_size'],
            'interpolation_method': 'linear',
            'use_sdf': interpolated_data['sdf_used'],
            'processing_time_seconds': duration,
            'description': '32x32x32网格插值结果，使用新的data_UNS目录结构快速验证'
        }

        storage.save(interpolated_data, output_h5, metadata=metadata)
        print(f"  [OK] HDF5文件已保存: {output_h5}")

        # 4. 转换为VTK格式
        print(f"\n4. 转换为VTK格式...")
        try:
            storage.convert_to_vtk(output_h5, output_vts)
            print(f"  [OK] VTK文件已保存: {output_vts}")
        except Exception as e:
            print(f"  [WARNING] VTK转换失败: {e}")

        print(f"\n=== 32x32x32插值验证成功完成 ===")
        print(f"输出文件:")
        print(f"  - HDF5: {output_h5}")
        print(f"  - VTK:  {output_vts}")
        print(f"处理时间: {duration:.1f}秒")

        return True

    except Exception as e:
        print(f"\n[ERROR] 插值失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)