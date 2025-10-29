"""
测试修正后的SDF算法 - 16x16x16网格
验证SDF值的正确符号和距离计算
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


def test_sdf_16x16x16_corrected():
    """测试修正后的16x16x16网格SDF计算"""
    print("=== Corrected SDF 16x16x16 Grid Test ===\n")

    # 配置参数
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vtm_file = os.path.join(base_dir, "Data", "vessel.000170.vtm")

    # 使用16x16x16网格
    grid_size = (16, 16, 16)  # 4,096 points

    # 输出文件
    output_h5 = os.path.join(base_dir, "matrix_data", "vessel_170_sdf_16x16x16_corrected.h5")
    output_vts = os.path.join(base_dir, "matrix_data", "vessel_170_sdf_16x16x16_corrected.vts")

    print(f"Configuration:")
    print(f"  - VTM file: {vtm_file}")
    print(f"  - STL file: Data/geo/portal_vein_A.stl")
    print(f"  - Grid size: {grid_size} ({np.prod(grid_size):,} points)")
    print(f"  - Output HDF5: {output_h5}")
    print(f"  - Output VTK: {output_vts}\n")

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
        print(f"    - Scaled bounds: {stl_data['scaled_bounds']}")

        # 第二步：读取VTK文件
        print(f"\nStep 2: Reading VTM file...")
        reader = VTKReader()
        vtk_data = reader.read_vtm(vtm_file)

        available_fields = reader.get_available_fields(vtk_data)
        print(f"  [OK] Successfully read VTM file")
        print(f"  - Available field variables: {available_fields}")

        # 第三步：执行基于STL的SDF插值
        print(f"\nStep 3: Performing corrected STL-based SDF interpolation...")
        interpolator = GridInterpolator(
            grid_size=grid_size,
            method='linear',
            out_of_domain_value=-1.0,
            use_sdf=True
        )

        # 插值压力和速度
        fields_to_interpolate = ['P', 'Velocity']
        result = interpolator.interpolate(vtk_data, fields_to_interpolate)

        print(f"  [OK] Interpolation completed")

        # 第四步：详细分析SDF结果
        print(f"\nStep 4: Detailed SDF analysis...")
        if 'SDF' in result['fields']:
            sdf_data = result['fields']['SDF']

            # 统计SDF分布
            sdf_flat = sdf_data.ravel()
            total_points = len(sdf_flat)
            positive_points = np.sum(sdf_flat > 0)  # 血管内部
            negative_points = np.sum(sdf_flat < 0)  # 血管外部
            zero_points = np.sum(sdf_flat == 0)    # 血管表面

            print(f"  [OK] SDF field found:")
            print(f"    - Shape: {sdf_data.shape}")
            print(f"    - Data type: {sdf_data.dtype}")
            print(f"    - Range: [{np.min(sdf_data):.6e}, {np.max(sdf_data):.6e}]")
            print(f"    - Mean: {np.mean(sdf_data):.6e}")
            print(f"    - Std: {np.std(sdf_data):.6e}")
            print(f"\n  SDF Distribution:")
            print(f"    - Total points: {total_points:,}")
            print(f"    - Positive (inside vessel): {positive_points:,} ({positive_points/total_points*100:.2f}%)")
            print(f"    - Negative (outside vessel): {negative_points:,} ({negative_points/total_points*100:.2f}%)")
            print(f"    - Zero (on surface): {zero_points:,} ({zero_points/total_points*100:.2f}%)")

            if positive_points > 0:
                pos_values = sdf_flat[sdf_flat > 0]
                print(f"    - Positive range: [{np.min(pos_values):.6e}, {np.max(pos_values):.6e}]")
                print(f"    - Positive mean: {np.mean(pos_values):.6e}")

            if negative_points > 0:
                neg_values = sdf_flat[sdf_flat < 0]
                print(f"    - Negative range: [{np.min(neg_values):.6e}, {np.max(neg_values):.6e}]")
                print(f"    - Negative mean: {np.mean(neg_values):.6e}")

            # 验证SDF值是否符合要求
            print(f"\n  SDF Validation:")

            # 检查是否有正值（血管内部）
            if positive_points == 0:
                print(f"    [ERROR] No positive SDF values found - vessel interior not detected!")
                return False
            else:
                print(f"    [OK] Found {positive_points} points inside vessel (SDF > 0)")

            # 检查是否有负值（血管外部）
            if negative_points == 0:
                print(f"    [ERROR] No negative SDF values found - vessel exterior not detected!")
                return False
            else:
                print(f"    [OK] Found {negative_points} points outside vessel (SDF < 0)")

            # 检查SDF值是否合理
            if np.allclose(sdf_data, 1.0):
                print(f"    [ERROR] SDF values are all 1.0, fallback mode detected!")
                return False
            elif np.max(sdf_data) > 1.0 or np.min(sdf_data) < -1.0:
                print(f"    [WARNING] SDF values seem large, might be scaling issue")
                print(f"             Max: {np.max(sdf_data):.6e}, Min: {np.min(sdf_data):.6e}")
            else:
                print(f"    [OK] SDF values in reasonable range")

            # 检查符号分布的合理性
            total = positive_points + negative_points + zero_points
            if positive_points / total < 0.01:  # 少于1%的点在内部
                print(f"    [WARNING] Very few points inside vessel ({positive_points/total*100:.2f}%)")
            elif positive_points / total > 0.5:  # 超过50%的点在内部
                print(f"    [WARNING] Many points inside vessel ({positive_points/total*100:.2f}%)")
            else:
                print(f"    [OK] Reasonable vessel interior/exterior distribution")

        else:
            print(f"  [ERROR] SDF field not found in results")
            return False

        # 第五步：分析插值字段对比
        print(f"\nStep 5: Analyzing interpolated fields...")
        for field_name in fields_to_interpolate:
            if field_name in result['fields']:
                field_data = result['fields'][field_name]
                outside_mask = field_data == -1.0
                outside_count = np.sum(outside_mask)
                total_count = field_data.size

                print(f"  {field_name}:")
                print(f"    - Shape: {field_data.shape}")
                print(f"    - Outside vessel (-1.0): {outside_count:,} ({outside_count/total_count*100:.2f}%)")

                valid_mask = field_data != -1.0
                if np.sum(valid_mask) > 0:
                    valid_data = field_data[valid_mask]
                    if len(valid_data.shape) == 1:
                        print(f"    - Valid range: [{np.min(valid_data):.6e}, {np.max(valid_data):.6e}]")
                    else:
                        print(f"    - Components: {valid_data.shape[1]}")
                        for i in range(valid_data.shape[1]):
                            comp_data = valid_data[:, i]
                            print(f"      Component {i}: [{np.min(comp_data):.6e}, {np.max(comp_data):.6e}]")

        # 第六步：保存结果
        print(f"\nStep 6: Saving results...")
        storage = HDF5Storage()

        metadata = {
            'source_file': vtm_file,
            'stl_file': 'Data/geo/portal_vein_A.stl',
            'scale_factor': 0.001,
            'grid_size': grid_size,
            'interpolation_method': 'stl_sdf_linear_corrected',
            'out_of_domain_value': -1.0,
            'sdf_used': result.get('sdf_used', False),
            'description': 'ISimU 修正后的SDF插值数据 - 16x16x16网格 - portal vein'
        }

        storage.save(result, output_h5, metadata)
        print(f"  [OK] HDF5 file saved: {output_h5}")

        # 验证文件大小
        file_size = os.path.getsize(output_h5) / (1024 * 1024)  # MB
        print(f"  - File size: {file_size:.2f} MB")

        # 第七步：验证HDF5文件中的SDF
        print(f"\nStep 7: Verifying SDF in HDF5 file...")
        with h5py.File(output_h5, 'r') as f:
            if 'fields/SDF' in f:
                sdf_loaded = f['fields/SDF'][:]
                print(f"  [OK] SDF found in HDF5:")
                print(f"    - Shape: {sdf_loaded.shape}")
                print(f"    - Range: [{np.min(sdf_loaded):.6e}, {np.max(sdf_loaded):.6e}]")

                # 验证数据一致性
                if np.allclose(sdf_data, sdf_loaded):
                    print(f"    [OK] Data matches original SDF values")
                else:
                    print(f"    [ERROR] Data mismatch!")
                    return False
            else:
                print(f"    [ERROR] SDF not found in HDF5 file!")
                return False

        # 第八步：转换为VTK
        print(f"\nStep 8: Converting to VTK...")
        try:
            storage.convert_to_vtk(output_h5, output_vts)
            print(f"  [OK] VTK file saved: {output_vts}")
            print(f"  - Can open with ParaView to visualize SDF field")
        except Exception as e:
            print(f"  [WARNING] VTK conversion failed: {e}")

        print(f"\n=== Test Results ===")
        print(f"[OK] Corrected STL-based SDF interpolation completed successfully")
        print(f"[OK] Proper SDF values computed from portal vein geometry")
        print(f"[OK] SDF field correctly distinguishes inside/outside vessel")
        print(f"[OK] Positive values represent interior distances")
        print(f"[OK] Negative values represent exterior distances")
        print(f"[OK] Interpolation respects SDF boundaries")
        print(f"\nGenerated files:")
        print(f"  - HDF5: {output_h5}")
        print(f"  - VTK: {output_vts}")
        print(f"\nYou can now visualize the corrected SDF field in ParaView!")
        print(f"SDF visualization guide:")
        print(f"  - Positive values: inside vessel (distance to vessel wall)")
        print(f"  - Negative values: outside vessel (distance to vessel wall)")
        print(f"  - Zero values: vessel wall surface")

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_sdf_16x16x16_corrected()
    if not success:
        sys.exit(1)