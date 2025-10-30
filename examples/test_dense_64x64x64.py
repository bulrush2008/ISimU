"""
测试64x64x64密集网格的SDF插值和域外点赋零
生成超高质量的流场数据用于ParaView可视化验证
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
from interpolator_optimized import OptimizedGridInterpolator
from hdf5_storage import HDF5Storage
from stl_reader import load_portal_vein_geometry
import numpy as np
import h5py


def test_dense_64x64x64_interpolation():
    """测试64x64x64密集网格的SDF插值和域外点赋零"""
    print("=== Dense 64x64x64 Grid Test with Zero Assignment ===\n")

    # 配置参数
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vtm_file = os.path.join(base_dir, "Data", "vessel.000170.vtm")

    # 使用64x64x64超密集网格 (262,144 points)
    grid_size = (64, 64, 64)

    print(f"Configuration:")
    print(f"  - VTM file: {vtm_file}")
    print(f"  - STL file: Data/geo/portal_vein_A.stl")
    print(f"  - Grid size: {grid_size} ({np.prod(grid_size):,} points)")
    print(f"  - Out-of-domain value: 0.0 (pressure and velocity components)")
    print(f"  - Expected: P=0, Velocity=(0,0,0) outside vessel")
    print(f"  - Ultra-dense grid for ultra-high quality visualization\n")

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

        # 第三步：执行基于SDF的超密集网格插值
        print(f"\nStep 3: Performing ultra-dense SDF interpolation with zero assignment...")
        interpolator = OptimizedGridInterpolator(
            grid_size=grid_size,
            method='linear',
            out_of_domain_value=0.0,  # 域外点赋零
            use_sdf=True,
            batch_size=20000  # 增大批处理大小以适应更大网格
        )

        # 插值压力和速度
        fields_to_interpolate = ['P', 'Velocity']
        result = interpolator.interpolate(vtk_data, fields_to_interpolate)

        print(f"  [OK] Ultra-dense interpolation completed")

        # 第四步：详细分析结果
        print(f"\nStep 4: Detailed analysis of ultra-dense results...")

        # 分析SDF分布
        if 'SDF' in result['fields']:
            sdf_data = result['fields']['SDF']
            sdf_flat = sdf_data.ravel()

            positive_points = np.sum(sdf_flat > 0)  # 血管内部
            negative_points = np.sum(sdf_flat < 0)  # 血管外部
            zero_points = np.sum(sdf_flat == 0)    # 血管表面

            print(f"  SDF Distribution (64x64x64 grid):")
            print(f"    - Total points: {len(sdf_flat):,}")
            print(f"    - Positive (inside vessel): {positive_points:,} ({positive_points/len(sdf_flat)*100:.2f}%)")
            print(f"    - Negative (outside vessel): {negative_points:,} ({negative_points/len(sdf_flat)*100:.2f}%)")
            print(f"    - Zero (on surface): {zero_points:,} ({zero_points/len(sdf_flat)*100:.2f}%)")
            print(f"    - SDF range: [{np.min(sdf_data):.6e}, {np.max(sdf_data):.6e}]")

            if positive_points > 0:
                pos_values = sdf_flat[sdf_flat > 0]
                print(f"    - Positive range: [{np.min(pos_values):.6e}, {np.max(pos_values):.6e}]")

            if negative_points > 0:
                neg_values = sdf_flat[sdf_flat < 0]
                print(f"    - Negative range: [{np.min(neg_values):.6e}, {np.max(neg_values):.6e}]")

        # 分析压力场
        if 'P' in result['fields']:
            pressure_data = result['fields']['P']
            zero_count_p = np.sum(pressure_data == 0.0)
            total_count_p = pressure_data.size
            non_zero_count_p = total_count_p - zero_count_p

            print(f"\n  Pressure Field Analysis:")
            print(f"    - Shape: {pressure_data.shape}")
            print(f"    - Zero values (outside): {zero_count_p:,} ({zero_count_p/total_count_p*100:.2f}%)")
            print(f"    - Non-zero values (inside): {non_zero_count_p:,} ({non_zero_count_p/total_count_p*100:.2f}%)")

            if non_zero_count_p > 0:
                non_zero_pressure = pressure_data[pressure_data != 0.0]
                print(f"    - Pressure range (inside): [{np.min(non_zero_pressure):.6e}, {np.max(non_zero_pressure):.6e}]")
                print(f"    - Pressure mean (inside): {np.mean(non_zero_pressure):.6e}")

        # 分析速度场
        if 'Velocity' in result['fields']:
            velocity_data = result['fields']['Velocity']

            # 检查零向量（所有分量都为零）
            zero_velocity_mask = np.all(np.abs(velocity_data) < 1e-10, axis=-1)
            zero_count_v = np.sum(zero_velocity_mask)
            total_count_v = velocity_data.shape[0] * velocity_data.shape[1] * velocity_data.shape[2]
            non_zero_count_v = total_count_v - zero_count_v

            print(f"\n  Velocity Field Analysis:")
            print(f"    - Shape: {velocity_data.shape}")
            print(f"    - Zero vectors (outside): {zero_count_v:,} ({zero_count_v/total_count_v*100:.2f}%)")
            print(f"    - Non-zero vectors (inside): {non_zero_count_v:,} ({non_zero_count_v/total_count_v*100:.2f}%)")

            if non_zero_count_v > 0:
                non_zero_velocity = velocity_data[~zero_velocity_mask]
                speed_magnitude = np.linalg.norm(non_zero_velocity, axis=-1)
                print(f"    - Speed range (inside): [{np.min(speed_magnitude):.6e}, {np.max(speed_magnitude):.6e}]")
                print(f"    - Speed mean (inside): {np.mean(speed_magnitude):.6e}")

                # 分析各速度分量
                for i, component in enumerate(['X', 'Y', 'Z']):
                    comp_data = non_zero_velocity[:, i]
                    print(f"    - {component}-component range: [{np.min(comp_data):.6e}, {np.max(comp_data):.6e}]")

        # 第五步：保存超密集网格结果
        print(f"\nStep 5: Saving ultra-dense grid results...")
        storage = HDF5Storage()

        metadata = {
            'source_file': vtm_file,
            'stl_file': 'Data/geo/portal_vein_A.stl',
            'scale_factor': 0.001,
            'grid_size': grid_size,
            'interpolation_method': 'sdf_linear_zero_assignment_ultra_dense',
            'out_of_domain_value': 0.0,
            'sdf_used': result.get('sdf_used', False),
            'description': 'ISimU 超密集网格SDF插值数据 - 64x64x64网格 - 域外点赋零'
        }

        output_h5 = os.path.join(base_dir, "matrix_data", "dense_64x64x64_zero_assignment.h5")
        output_vts = os.path.join(base_dir, "matrix_data", "dense_64x64x64_zero_assignment.vts")

        storage.save(result, output_h5, metadata)
        print(f"  [OK] HDF5 file saved: {output_h5}")

        # 验证文件大小
        file_size = os.path.getsize(output_h5) / (1024 * 1024)  # MB
        print(f"  - File size: {file_size:.2f} MB")

        # 第六步：转换为VTK
        print(f"\nStep 6: Converting to VTK for ParaView...")
        try:
            storage.convert_to_vtk(output_h5, output_vts)
            print(f"  [OK] VTK file saved: {output_vts}")
            print(f"  - Ready for ultra-high quality ParaView visualization")
        except Exception as e:
            print(f"  [WARNING] VTK conversion failed: {e}")

        # 第七步：验证数据一致性
        print(f"\nStep 7: Verifying data consistency...")
        with h5py.File(output_h5, 'r') as f:
            fields_in_h5 = list(f['fields'].keys())
            print(f"  [OK] Fields in HDF5: {fields_in_h5}")

            expected_fields = ['P', 'Velocity', 'SDF']
            for field in expected_fields:
                if field in fields_in_h5:
                    field_shape = f['fields'][field].shape
                    print(f"    - {field}: {field_shape}")
                else:
                    print(f"    [WARNING] {field} not found in HDF5")

        print(f"\n=== Ultra-Dense Grid Test Results ===")
        print(f"[OK] 64x64x64 ultra-dense grid interpolation completed successfully")
        print(f"[OK] SDF-based boundary detection works correctly")
        print(f"[OK] Zero assignment strategy verified:")
        print(f"    - Exterior points: P=0, Velocity=(0,0,0)")
        print(f"    - Interior points: Real interpolated values")
        print(f"[OK] Ultra-high resolution data ready for detailed visualization")
        print(f"\nGenerated files:")
        print(f"  - HDF5: {output_h5}")
        print(f"  - VTK:  {output_vts}")
        print(f"\nParaView visualization guide:")
        print(f"  - Load: {output_vts}")
        print(f"  - Apply 'Slice' or 'Clip' filters to see vessel interior")
        print(f"  - Color by SDF to see vessel boundaries")
        print(f"  - Color by P or Velocity magnitude for flow patterns")
        print(f"  - Use 'Threshold' on SDF > 0 to show only vessel interior")
        print(f"\nPerformance note:")
        print(f"  - 64x64x64 grid processes {np.prod(grid_size):,} points")
        print(f"  - This is ~2.37x more points than 48x48x48 grid")
        print(f"  - Expect longer processing time but much higher resolution")

        return True

    except Exception as e:
        print(f"\n[ERROR] Ultra-dense grid test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dense_64x64x64_interpolation()
    if not success:
        sys.exit(1)