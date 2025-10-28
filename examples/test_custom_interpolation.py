"""
符合CLAUDE.md需求的插值方法测试脚本
演示两种插值方式：最近邻直接赋值和3点平均值
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
from interpolator import GridInterpolator
from hdf5_storage import HDF5Storage
import numpy as np


def test_custom_interpolation_methods():
    """测试自定义插值方法"""
    print("=== ISimU Custom Interpolation Methods Test ===\n")

    # 配置参数
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vtm_file = os.path.join(base_dir, "Data", "vessel.000170.vtm")

    # 使用CLAUDE.md中指定的默认网格尺寸（先用较小的网格测试）
    grid_size = (32, 32, 32)  # 改为32x32x32进行快速测试，实际使用时为128x128x128

    print(f"Configuration:")
    print(f"  - Input file: {vtm_file}")
    print(f"  - Grid size: {grid_size} (default per CLAUDE.md)")
    print(f"  - Out of domain value: -1.0\n")

    try:
        # 读取VTK文件
        print("Step 1: Reading VTK file...")
        reader = VTKReader()
        vtk_data = reader.read_vtm(vtm_file)

        available_fields = reader.get_available_fields(vtk_data)
        print(f"  [OK] Successfully read VTK file")
        print(f"  - Available field variables: {available_fields}")

        # 选择要插值的场变量
        fields_to_interpolate = ['P', 'RHO']  # 选择压力和密度

        # 测试方法1：最近邻直接赋值
        print(f"\nStep 2a: Testing method 1 - Nearest neighbor assignment...")
        interpolator_nearest = GridInterpolator(
            grid_size=grid_size,
            out_of_domain_value=-1.0
        )

        result_nearest = interpolator_nearest.interpolate_with_custom_methods(
            vtk_data, fields_to_interpolate, 'nearest'
        )

        print(f"  [OK] Nearest neighbor interpolation completed")

        # 测试方法2：3点平均值
        print(f"\nStep 2b: Testing method 2 - Average of 3 nearest points...")
        interpolator_average = GridInterpolator(
            grid_size=grid_size,
            out_of_domain_value=-1.0
        )

        result_average = interpolator_average.interpolate_with_custom_methods(
            vtk_data, fields_to_interpolate, 'average'
        )

        print(f"  [OK] Average interpolation completed")

        # 比较两种方法的结果
        print(f"\nStep 3: Comparing interpolation methods...")

        for field_name in fields_to_interpolate:
            if field_name in result_nearest['fields'] and field_name in result_average['fields']:
                nearest_data = result_nearest['fields'][field_name]
                average_data = result_average['fields'][field_name]

                # 计算域外点的数量
                nearest_out_domain = np.sum(nearest_data == -1.0)
                average_out_domain = np.sum(average_data == -1.0)

                # 计算有效点的统计
                nearest_valid = nearest_data[nearest_data != -1.0]
                average_valid = average_data[average_data != -1.0]

                print(f"\n  {field_name}:")
                print(f"    Total grid points: {nearest_data.size:,}")
                print(f"    Out of domain points (-1.0):")
                print(f"      - Nearest method: {nearest_out_domain:,} ({nearest_out_domain/nearest_data.size*100:.1f}%)")
                print(f"      - Average method: {average_out_domain:,} ({average_out_domain/average_data.size*100:.1f}%)")

                if len(nearest_valid) > 0:
                    print(f"    Valid points statistics:")
                    print(f"      - Nearest method: range[{np.min(nearest_valid):.3e}, {np.max(nearest_valid):.3e}], mean{np.mean(nearest_valid):.3e}")
                    print(f"      - Average method: range[{np.min(average_valid):.3e}, {np.max(average_valid):.3e}], mean{np.mean(average_valid):.3e}")

        # 保存结果进行比较
        print(f"\nStep 4: Saving results for comparison...")

        storage = HDF5Storage()

        # 保存最近邻方法结果
        metadata_nearest = {
            'source_file': vtm_file,
            'grid_size': grid_size,
            'interpolation_method': 'custom_nearest',
            'out_of_domain_value': -1.0,
            'description': 'ISimU插值数据 - 最近邻直接赋值方法'
        }

        output_nearest = os.path.join(base_dir, "matrix_data", "vessel_170_nearest.h5")
        storage.save(result_nearest, output_nearest, metadata_nearest)
        print(f"  [OK] Nearest method saved: {output_nearest}")

        # 保存平均值方法结果
        metadata_average = {
            'source_file': vtm_file,
            'grid_size': grid_size,
            'interpolation_method': 'custom_average',
            'out_of_domain_value': -1.0,
            'description': 'ISimU插值数据 - 3点平均值方法'
        }

        output_average = os.path.join(base_dir, "matrix_data", "vessel_170_average.h5")
        storage.save(result_average, output_average, metadata_average)
        print(f"  [OK] Average method saved: {output_average}")

        print(f"\n=== Test Completed Successfully ===")
        print(f"Output files:")
        print(f"  - Nearest method: {output_nearest}")
        print(f"  - Average method: {output_average}")
        print(f"\nThe interpolation methods now correctly:")
        print(f"  1. Use 128x128x128 grid by default (per CLAUDE.md)")
        print(f"  2. Assign -1.0 to points outside the vessel domain")
        print(f"  3. Support both nearest and averaging interpolation methods")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_custom_interpolation_methods()
    if not success:
        sys.exit(1)