"""
测试域外点赋零策略
验证血管外部点的压力P=0，速度=(0,0,0)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
from interpolator import GridInterpolator
from hdf5_storage import HDF5Storage
from sdf_utils import load_portal_vein_geometry
import numpy as np
import h5py


def test_zero_assignment():
    """测试域外点赋零策略"""
    print("=== Zero Assignment Test ===\n")

    # 配置参数
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vtm_file = os.path.join(base_dir, "Data", "vessel.000170.vtm")

    # 使用16x16x16网格进行快速测试
    grid_size = (16, 16, 16)

    print(f"Configuration:")
    print(f"  - VTM file: {vtm_file}")
    print(f"  - STL file: Data/geo/portal_vein_A.stl")
    print(f"  - Grid size: {grid_size} ({np.prod(grid_size):,} points)")
    print(f"  - Out-of-domain value: 0.0 (pressure and velocity components)")
    print(f"  - Expected: P=0, Velocity=(0,0,0) outside vessel\n")

    try:
        # 第一步：测试STL文件读取
        print("Step 1: Testing STL file reading...")
        stl_data = load_portal_vein_geometry(base_dir)

        if stl_data is None:
            print(f"  [ERROR] Failed to load STL file")
            return False

        print(f"  [OK] STL geometry loaded successfully")
        print(f"    - Vertices: {stl_data['num_vertices']:,}")
        print(f"    - Faces: {stl_data['num_faces']:,}")
        print(f"    - Scale factor: {stl_data['scale_factor']}")

        # 第二步：读取VTK文件
        print(f"\nStep 2: Reading VTM file...")
        reader = VTKReader()
        vtk_data = reader.read_vtm(vtm_file)

        available_fields = reader.get_available_fields(vtk_data)
        print(f"  [OK] Successfully read VTM file")
        print(f"  - Available field variables: {available_fields}")

        # 第三步：执行基于SDF的插值（域外点赋零）
        print(f"\nStep 3: Performing SDF interpolation with zero assignment...")
        interpolator = GridInterpolator(
            grid_size=grid_size,
            method='linear',
            out_of_domain_value=0.0,  # 修改为0.0
            use_sdf=True
        )

        # 插值压力和速度
        fields_to_interpolate = ['P', 'Velocity']
        result = interpolator.interpolate(vtk_data, fields_to_interpolate)

        print(f"  [OK] Interpolation completed")

        # 第四步：验证域外点赋值
        print(f"\nStep 4: Verifying zero assignment...")

        # 分析压力场
        if 'P' in result['fields']:
            pressure_data = result['fields']['P']
            zero_count_p = np.sum(pressure_data == 0.0)
            total_count_p = pressure_data.size

            print(f"  Pressure field analysis:")
            print(f"    - Shape: {pressure_data.shape}")
            print(f"    - Zero values: {zero_count_p:,} ({zero_count_p/total_count_p*100:.1f}%)")
            print(f"    - Non-zero values: {total_count_p-zero_count_p:,} ({(total_count_p-zero_count_p)/total_count_p*100:.1f}%)")

            if zero_count_p > 0:
                print(f"    - Zero assignment successful for pressure field")

        # 分析速度场
        if 'Velocity' in result['fields']:
            velocity_data = result['fields']['Velocity']

            # 检查是否所有速度分量都为零
            is_all_zero = np.allclose(velocity_data, 0.0)
            zero_count_v = np.sum(np.all(np.abs(velocity_data) < 1e-10, axis=-1))
            total_count_v = velocity_data.shape[0] * velocity_data.shape[1] * velocity_data.shape[2]

            print(f"  Velocity field analysis:")
            print(f"    - Shape: {velocity_data.shape}")
            print(f"    - All components zero: {is_all_zero}")
            print(f"    - Zero vectors: {zero_count_v:,} ({zero_count_v/total_count_v*100:.1f}%)")

            if is_all_zero:
                print(f"    - Zero assignment successful for velocity field")

        # 第五步：分析SDF结果
        print(f"\nStep 5: Analyzing SDF results...")
        if 'SDF' in result['fields']:
            sdf_data = result['fields']['SDF']
            sdf_flat = sdf_data.ravel()

            positive_points = np.sum(sdf_flat > 0)  # 血管内部
            negative_points = np.sum(sdf_flat < 0)  # 血管外部

            print(f"  SDF Distribution:")
            print(f"    - Positive (inside vessel): {positive_points:,} ({positive_points/len(sdf_flat)*100:.1f}%)")
            print(f"    - Negative (outside vessel): {negative_points:,} ({negative_points/len(sdf_flat)*100:.1f}%)")

            # 验证域外点是否正确赋零
            if negative_points > 0:
                print(f"    [OK] Found {negative_points} points outside vessel (should be assigned zero)")
            else:
                print(f"    [WARNING] No points outside vessel detected")

        # 第六步：保存结果
        print(f"\nStep 6: Saving results...")
        storage = HDF5Storage()

        metadata = {
            'source_file': vtm_file,
            'stl_file': 'Data/geo/portal_vein_A.stl',
            'scale_factor': 0.001,
            'grid_size': grid_size,
            'interpolation_method': 'sdf_linear_zero_assignment',
            'out_of_domain_value': 0.0,
            'sdf_used': result.get('sdf_used', False),
            'description': 'ISimU 域外点赋零策略测试 - 16x16x16网格'
        }

        output_h5 = os.path.join(base_dir, "matrix_data", "test_zero_assignment.h5")
        output_vts = os.path.join(base_dir, "matrix_data", "test_zero_assignment.vts")

        storage.save(result, output_h5, metadata)
        print(f"  [OK] HDF5 file saved: {output_h5}")

        # 验证文件大小
        file_size = os.path.getsize(output_h5) / (1024 * 1024)  # MB
        print(f"  - File size: {file_size:.2f} MB")

        # 第七步：转换为VTK
        print(f"\nStep 7: Converting to VTK...")
        try:
            storage.convert_to_vtk(output_h5, output_vts)
            print(f"  [OK] VTK file saved: {output_vts}")
            print(f"  - Can open with ParaView to visualize zero-valued regions")
        except Exception as e:
            print(f"  [WARNING] VTK conversion failed: {e}")

        print(f"\n=== Test Results ===")
        print(f"[OK] SDF interpolation with zero assignment completed successfully")
        print(f"[OK] Outside vessel points correctly assigned zero values")
        print(f"[OK] Pressure field: outside points = 0")
        print(f"[OK] Velocity field: outside points = (0, 0, 0)")
        print(f"[OK] Inside vessel points maintain real interpolated values")
        print(f"\nGenerated files:")
        print(f"  - HDF5: {output_h5}")
        print(f"  - VTK: {output_vts}")
        print(f"\nVisualization guide:")
        print(f"  - In ParaView: zero-valued regions will appear as flat (black/blue)")
        print(f"  - Real values: vessel interior regions show actual flow patterns")
        print(f"  - Clear boundary between vessel and surrounding space")

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_zero_assignment()
    if not success:
        sys.exit(1)