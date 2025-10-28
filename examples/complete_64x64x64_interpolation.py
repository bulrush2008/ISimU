"""
完整的64x64x64网格插值示例
插值字段：压力P、速度Velocity、CellID
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
from interpolator import GridInterpolator
from hdf5_storage import HDF5Storage
import numpy as np


def main():
    """执行完整的64x64x64网格插值"""
    print("=== ISimU Complete 64x64x64 Grid Interpolation ===\n")

    # 配置参数
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vtm_file = os.path.join(base_dir, "Data", "vessel.000170.vtm")

    # 64x64x64网格配置
    grid_size = (64, 64, 64)

    # 输出文件路径
    output_h5 = os.path.join(base_dir, "matrix_data", "vessel_170_64x64x64_complete.h5")
    output_vts = os.path.join(base_dir, "matrix_data", "vessel_170_64x64x64_complete.vts")

    print(f"Configuration:")
    print(f"  - Input file: {vtm_file}")
    print(f"  - Grid size: {grid_size}")
    print(f"  - Fields to interpolate: P, Velocity, CellID")
    print(f"  - SDF enabled: Yes")
    print(f"  - Out of domain value: -1.0")
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

        # 检查面片数据
        has_faces = False
        for block in vtk_data['blocks']:
            if 'faces' in block and block['faces'] is not None:
                has_faces = True
                print(f"  - Found surface faces: {len(block['faces'])} triangles")
                break

        if not has_faces:
            print(f"  [WARNING] No surface faces found, using fallback interpolation")

        # 第二步：执行插值
        print(f"\nStep 2: Performing grid interpolation...")

        # 创建插值器
        interpolator = GridInterpolator(
            grid_size=grid_size,
            method='linear',
            out_of_domain_value=-1.0,
            use_sdf=True
        )

        # 要插值的字段
        fields_to_interpolate = ['P', 'Velocity', 'CellID']
        print(f"  - Interpolating fields: {fields_to_interpolate}")

        # 执行插值
        interpolated_data = interpolator.interpolate(vtk_data, fields_to_interpolate)

        print(f"  [OK] Interpolation completed")

        # 第三步：分析插值结果
        print(f"\nStep 3: Analyzing interpolation results...")

        total_points = grid_size[0] * grid_size[1] * grid_size[2]
        print(f"  - Total grid points: {total_points:,}")

        if interpolated_data.get('sdf_used', False):
            inside_count = interpolated_data.get('inside_point_count', 0)
            outside_count = interpolated_data.get('outside_point_count', 0)
            print(f"  - Points inside vessel: {inside_count:,} ({inside_count/total_points*100:.1f}%)")
            print(f"  - Points outside vessel: {outside_count:,} ({outside_count/total_points*100:.1f}%)")
        else:
            print(f"  - SDF not used, all points processed")

        # 分析每个字段
        for field_name in fields_to_interpolate:
            if field_name in interpolated_data['fields']:
                field_data = interpolated_data['fields'][field_name]

                # 统计域外点
                outside_mask = field_data == -1.0
                outside_count = np.sum(outside_mask)

                # 统计有效点
                valid_mask = field_data != -1.0
                valid_data = field_data[valid_mask]

                print(f"\n  {field_name}:")
                print(f"    - Shape: {field_data.shape}")
                print(f"    - Outside vessel (-1.0): {outside_count:,} ({outside_count/field_data.size*100:.1f}%)")

                if len(valid_data) > 0:
                    if len(valid_data.shape) == 1:
                        # 标量场
                        print(f"    - Valid range: [{np.min(valid_data):.3e}, {np.max(valid_data):.3e}]")
                        print(f"    - Valid mean: {np.mean(valid_data):.3e}")
                        print(f"    - Valid std: {np.std(valid_data):.3e}")
                    else:
                        # 矢量场
                        print(f"    - Components: {valid_data.shape[1]}")
                        for i in range(valid_data.shape[1]):
                            component_data = valid_data[:, i]
                            print(f"    - Component {i}: range[{np.min(component_data):.3e}, {np.max(component_data):.3e}], mean{np.mean(component_data):.3e}")

        # 第四步：保存为HDF5格式
        print(f"\nStep 4: Saving as HDF5 format...")
        storage = HDF5Storage()

        # 准备元数据
        metadata = {
            'source_file': vtm_file,
            'grid_size': grid_size,
            'interpolation_method': 'sdf_linear',
            'out_of_domain_value': -1.0,
            'sdf_used': interpolated_data.get('sdf_used', False),
            'interpolated_fields': fields_to_interpolate,
            'description': 'ISimU 64x64x64完整插值数据 - 压力、速度、CellID'
        }

        storage.save(interpolated_data, output_h5, metadata)
        print(f"  [OK] HDF5 file saved: {output_h5}")

        # 验证保存的文件
        file_info = storage.get_file_info(output_h5)
        print(f"  - File size: {file_info['total_data_size_mb']:.2f} MB")

        # 第五步：转换为VTK格式用于可视化
        print(f"\nStep 5: Converting to VTK format...")
        try:
            storage.convert_to_vtk(output_h5, output_vts)
            print(f"  [OK] VTK file saved: {output_vts}")
            print(f"  - Can open with ParaView: {output_vts}")
        except Exception as e:
            print(f"  [WARNING] VTK conversion failed: {e}")

        # 第六步：数据验证
        print(f"\nStep 6: Data validation...")
        try:
            # 重新加载数据进行验证
            loaded_data = storage.load(output_h5)
            print(f"  [OK] Successfully reloaded HDF5 file")

            # 验证数据完整性
            for field_name in fields_to_interpolate:
                if field_name in loaded_data['fields']:
                    field_data = loaded_data['fields'][field_name]
                    nan_count = np.sum(np.isnan(field_data))
                    inf_count = np.sum(np.isinf(field_data))

                    print(f"  - {field_name}:")
                    print(f"    Shape: {field_data.shape}")
                    print(f"    NaN points: {nan_count}")
                    print(f"    Infinite points: {inf_count}")

                    if nan_count == 0 and inf_count == 0:
                        print(f"    [OK] Data quality is good")
                    else:
                        print(f"    [WARNING] Data quality issues detected")

        except Exception as e:
            print(f"  [WARNING] Validation failed: {e}")

        print(f"\n=== Interpolation Completed Successfully ===")
        print(f"Summary:")
        print(f"  - Grid size: {grid_size}")
        print(f"  - Interpolated fields: {fields_to_interpolate}")
        print(f"  - SDF used: {interpolated_data.get('sdf_used', False)}")
        print(f"  - Output files:")
        print(f"    * HDF5 data: {output_h5}")
        print(f"    * VTK visualization: {output_vts}")
        print(f"\nAll three fields (P, Velocity, CellID) have been successfully interpolated!")

        return True

    except Exception as e:
        print(f"\n[ERROR] Interpolation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)