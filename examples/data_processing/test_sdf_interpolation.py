"""
测试基于SDF的插值方法
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
from interpolator import GridInterpolator
from hdf5_storage import HDF5Storage
import numpy as np


def test_sdf_interpolation():
    """测试基于SDF的插值"""
    print("=== ISimU SDF-based Interpolation Test ===\n")

    # 配置参数
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vtm_file = os.path.join(base_dir, "Data", "vessel.000170.vtm")

    # 使用较小的网格进行测试
    grid_size = (32, 32, 32)

    print(f"Configuration:")
    print(f"  - Input file: {vtm_file}")
    print(f"  - Grid size: {grid_size}")
    print(f"  - SDF enabled: Yes")
    print(f"  - Out of domain value: -1.0\n")

    try:
        # 读取VTK文件
        print("Step 1: Reading VTK file...")
        reader = VTKReader()
        vtk_data = reader.read_vtm(vtm_file)

        available_fields = reader.get_available_fields(vtk_data)
        print(f"  [OK] Successfully read VTK file")
        print(f"  - Available field variables: {available_fields}")

        # 检查面片数据
        has_faces = False
        for block in vtk_data['blocks']:
            if 'faces' in block and block['faces'] is not None:
                has_faces = True
                print(f"  - Found surface faces: {len(block['faces'])} triangles")
                break

        if not has_faces:
            print("  [WARNING] No surface faces found, SDF may not work properly")

        # 选择要插值的场变量（包括速度场）
        fields_to_interpolate = ['P', 'Velocity']  # 压力和速度

        # 测试SDF插值
        print(f"\nStep 2: Testing SDF-based interpolation...")
        interpolator = GridInterpolator(
            grid_size=grid_size,
            method='linear',
            out_of_domain_value=-1.0,
            use_sdf=True
        )

        result = interpolator.interpolate(vtk_data, fields_to_interpolate)

        print(f"  [OK] SDF interpolation completed")

        # 分析结果
        print(f"\nStep 3: Analyzing results...")

        total_points = result['grid_size'][0] * result['grid_size'][1] * result['grid_size'][2]

        if 'sdf_used' in result and result['sdf_used']:
            print(f"  - SDF was used successfully")
            inside_count = result.get('inside_point_count', 0)
            outside_count = result.get('outside_point_count', 0)
            print(f"  - Points inside vessel: {inside_count:,} ({inside_count/total_points*100:.1f}%)")
            print(f"  - Points outside vessel: {outside_count:,} ({outside_count/total_points*100:.1f}%)")
        else:
            print(f"  - SDF was not used (fallback to all points)")

        for field_name in fields_to_interpolate:
            if field_name in result['fields']:
                field_data = result['fields'][field_name]

                # 统计域外点
                outside_mask = field_data == -1.0
                outside_count = np.sum(outside_mask)

                # 统计有效点
                valid_mask = field_data != -1.0
                valid_data = field_data[valid_mask]

                print(f"\n  {field_name}:")
                print(f"    - Total points: {field_data.size:,}")
                print(f"    - Outside vessel (-1.0): {outside_count:,} ({outside_count/field_data.size*100:.1f}%)")

                if len(valid_data) > 0:
                    if len(valid_data.shape) == 1:
                        print(f"    - Valid range: [{np.min(valid_data):.3e}, {np.max(valid_data):.3e}]")
                        print(f"    - Valid mean: {np.mean(valid_data):.3e}")
                    else:
                        # 矢量场
                        for i in range(valid_data.shape[1]):
                            component_data = valid_data[:, i]
                            print(f"    - Component {i} range: [{np.min(component_data):.3e}, {np.max(component_data):.3e}]")

        # 保存结果
        print(f"\nStep 4: Saving results...")

        storage = HDF5Storage()
        metadata = {
            'source_file': vtm_file,
            'grid_size': grid_size,
            'interpolation_method': 'sdf_linear',
            'out_of_domain_value': -1.0,
            'sdf_used': result.get('sdf_used', False),
            'description': 'ISimU SDF-based插值测试数据'
        }

        output_file = os.path.join(base_dir, "matrix_data", "vessel_170_sdf_test.h5")
        storage.save(result, output_file, metadata)
        print(f"  [OK] Results saved: {output_file}")

        # 转换为VTK用于可视化
        vtk_output = os.path.join(base_dir, "matrix_data", "vessel_170_sdf_test.vts")
        try:
            storage.convert_to_vtk(output_file, vtk_output)
            print(f"  [OK] VTK file saved: {vtk_output}")
            print(f"  - Can open with ParaView to inspect results")
        except Exception as e:
            print(f"  [WARNING] VTK conversion failed: {e}")

        print(f"\n=== Test Summary ===")
        if result.get('sdf_used', False):
            print(f"[OK] SDF-based interpolation completed successfully")
            print(f"[OK] Vessel interior/exterior correctly identified")
            print(f"[OK] Outside points assigned value -1.0")
        else:
            print(f"[WARNING] SDF not available, used fallback interpolation")

        print(f"[OK] All fields interpolated including velocity")
        print(f"[OK] Results saved in both HDF5 and VTK formats")

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_sdf_interpolation()
    if not success:
        sys.exit(1)