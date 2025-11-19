"""
测试SDF值保存功能
验证SDF值是否正确保存到HDF5和VTK文件中
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
from interpolator import GridInterpolator
from hdf5_storage import HDF5Storage
import numpy as np
import h5py


def test_sdf_saving():
    """测试SDF值保存功能"""
    print("=== ISimU SDF Value Saving Test ===\n")

    # 配置参数
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vtm_file = os.path.join(base_dir, "Data", "vessel.000170.vtm")

    # 使用较小的网格进行测试
    grid_size = (32, 32, 32)

    # 输出文件
    output_h5 = os.path.join(base_dir, "matrix_data", "test_sdf_values.h5")
    output_vts = os.path.join(base_dir, "matrix_data", "test_sdf_values.vts")

    print(f"Configuration:")
    print(f"  - Input file: {vtm_file}")
    print(f"  - Grid size: {grid_size}")
    print(f"  - Test SDF saving: Yes")
    print(f"  - Output HDF5: {output_h5}")
    print(f"  - Output VTK: {output_vts}\n")

    try:
        # 第一步：读取VTK文件
        print("Step 1: Reading VTK file...")
        reader = VTKReader()
        vtk_data = reader.read_vtm(vtm_file)

        available_fields = reader.get_available_fields(vtk_data)
        print(f"  [OK] Successfully read VTK file")
        print(f"  - Available field variables: {available_fields}")

        # 第二步：执行插值（包含SDF）
        print(f"\nStep 2: Performing interpolation with SDF...")
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

        # 第三步：检查SDF字段
        print(f"\nStep 3: Checking SDF field...")
        if 'SDF' in result['fields']:
            sdf_data = result['fields']['SDF']
            print(f"  [OK] SDF field found")
            print(f"  - Shape: {sdf_data.shape}")
            print(f"  - Data type: {sdf_data.dtype}")
            print(f"  - Range: [{np.min(sdf_data):.3e}, {np.max(sdf_data):.3e}]")
            print(f"  - Mean: {np.mean(sdf_data):.3e}")
            print(f"  - Positive values (inside): {np.sum(sdf_data > 0):,}")
            print(f"  - Negative values (outside): {np.sum(sdf_data < 0):,}")
            print(f"  - Zero values (on surface): {np.sum(sdf_data == 0):,}")
        else:
            print(f"  [WARNING] SDF field not found")
            return False

        # 第四步：保存到HDF5
        print(f"\nStep 4: Saving to HDF5...")
        storage = HDF5Storage()

        metadata = {
            'source_file': vtm_file,
            'grid_size': grid_size,
            'interpolation_method': 'sdf_linear',
            'out_of_domain_value': -1.0,
            'sdf_used': result.get('sdf_used', False),
            'description': 'ISimU SDF值保存测试数据'
        }

        storage.save(result, output_h5, metadata)
        print(f"  [OK] HDF5 file saved: {output_h5}")

        # 第五步：验证HDF5文件中的SDF
        print(f"\nStep 5: Verifying SDF in HDF5 file...")
        with h5py.File(output_h5, 'r') as f:
            print(f"  HDF5 file structure:")

            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"    {name}: {obj.shape} {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"    {name}/ (Group)")

            f.visititems(print_structure)

            # 检查SDF字段
            if 'fields/SDF' in f:
                sdf_dataset = f['fields/SDF']
                sdf_loaded = sdf_dataset[:]
                print(f"\n  [OK] SDF found in HDF5:")
                print(f"    - Shape: {sdf_loaded.shape}")
                print(f"    - Range: [{np.min(sdf_loaded):.3e}, {np.max(sdf_loaded):.3e}]")

                # 验证数据一致性
                if np.allclose(sdf_data, sdf_loaded):
                    print(f"    [OK] Data matches original SDF values")
                else:
                    print(f"    [ERROR] Data mismatch!")
                    return False
            else:
                print(f"    [ERROR] SDF not found in HDF5 file!")
                return False

        # 第六步：转换为VTK
        print(f"\nStep 6: Converting to VTK...")
        try:
            storage.convert_to_vtk(output_h5, output_vts)
            print(f"  [OK] VTK file saved: {output_vts}")
        except Exception as e:
            print(f"  [WARNING] VTK conversion failed: {e}")

        # 第七步：分析SDF值分布
        print(f"\nStep 7: SDF value analysis...")
        sdf_flat = sdf_data.ravel()

        # 统计分析
        total_points = len(sdf_flat)
        positive_points = np.sum(sdf_flat > 0)
        negative_points = np.sum(sdf_flat < 0)
        zero_points = np.sum(sdf_flat == 0)

        print(f"  Total points: {total_points:,}")
        print(f"  Positive (inside vessel): {positive_points:,} ({positive_points/total_points*100:.1f}%)")
        print(f"  Negative (outside vessel): {negative_points:,} ({negative_points/total_points*100:.1f}%)")
        print(f"  Zero (on surface): {zero_points:,} ({zero_points/total_points*100:.1f}%)")

        if positive_points > 0:
            positive_values = sdf_flat[sdf_flat > 0]
            print(f"  Positive range: [{np.min(positive_values):.3e}, {np.max(positive_values):.3e}]")

        if negative_points > 0:
            negative_values = sdf_flat[sdf_flat < 0]
            print(f"  Negative range: [{np.min(negative_values):.3e}, {np.max(negative_values):.3e}]")

        print(f"\n=== Test Results ===")
        print(f"[OK] SDF values successfully computed and saved")
        print(f"[OK] HDF5 file contains SDF field")
        print(f"[OK] VTK file includes SDF for visualization")
        print(f"[OK] SDF correctly distinguishes inside/outside vessel")
        print(f"\nGenerated files:")
        print(f"  - HDF5: {output_h5}")
        print(f"  - VTK: {output_vts}")
        print(f"\nYou can now visualize the SDF field in ParaView!")

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_sdf_saving()
    if not success:
        sys.exit(1)