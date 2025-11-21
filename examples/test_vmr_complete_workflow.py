"""
测试VMR数据的完整工作流程：从VMR配置到HDF5/VTS输出
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vmr_data_loader import VMRDataLoader
from interpolator_optimized import OptimizedGridInterpolator
from hdf5_storage import HDF5Storage
import numpy as np
import time


def test_vmr_complete_workflow():
    """测试VMR数据的完整工作流程"""
    print("=== VMR完整工作流程测试 ===")
    print("流程: VMR配置 → VTP/VTU读取 → SDF计算 → 插值 → HDF5/VTS输出")

    try:
        # 参数设置
        case_name = "0007_H_AO_H"
        grid_size = (48, 48, 48)  # 中等密度网格
        output_dir = "matrix_data_vmr"

        # 1. 创建VMR数据加载器
        print(f"\n1. 创建VMR数据加载器")
        loader = VMRDataLoader()
        print(f"  [OK] VMR数据加载器创建成功")

        # 2. 为插值器准备数据
        print(f"\n2. 准备插值器数据: {case_name}")
        start_time = time.time()

        interpolator_data = loader.create_interpolator_data(
            case_name=case_name,
            grid_size=grid_size,
            fields=['pressure', 'velocity']
        )

        if interpolator_data is None:
            print(f"  [ERROR] VMR数据准备失败")
            return False

        data_prep_time = time.time() - start_time
        print(f"  [OK] 数据准备完成 ({data_prep_time:.1f}s)")
        print(f"    - 顶点数: {interpolator_data['num_points']:,}")
        print(f"    - 物理场: {list(interpolator_data['point_data'].keys())}")

        # 3. 创建优化的插值器
        print(f"\n3. 创建优化插值器")
        interpolator = OptimizedGridInterpolator(
            grid_size=grid_size,
            use_sdf=True
        )
        print(f"  [OK] 优化插值器创建成功")

        # 4. 执行插值
        print(f"\n4. 执行插值计算")
        start_time = time.time()

        # 转换为插值器期望的格式
        vtk_compatible_data = {
            'type': 'UnstructuredGrid',
            'blocks': [interpolator_data]
        }

        result = interpolator.interpolate(vtk_compatible_data, fields=['pressure', 'velocity'])

        interpolation_time = time.time() - start_time

        if result is None:
            print(f"  [ERROR] 插值失败")
            return False

        print(f"  [OK] 插值完成 ({interpolation_time:.1f}s)")
        print(f"    - 网格尺寸: {result['grid_size']}")
        print(f"    - 内部点: {result['inside_point_count']:,}")
        print(f"    - 外部点: {result['outside_point_count']:,}")
        print(f"    - 插值字段: {list(result['fields'].keys())}")

        # 5. 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 6. 保存为HDF5格式
        print(f"\n5. 保存HDF5格式")
        start_time = time.time()

        hdf5_filename = f"{output_dir}/vmr_{case_name}_{grid_size[0]}x{grid_size[1]}x{grid_size[2]}.h5"
        writer = HDF5Storage()

        # 准备元数据
        metadata = {
            'case_name': case_name,
            'grid_size': grid_size,
            'interpolation_method': interpolator.method,
            'sdf_used': result['sdf_used'],
            'processing_times': {
                'data_preparation': data_prep_time,
                'interpolation': interpolation_time
            },
            'data_sources': {
                'geometry': interpolator_data.get('case_name', 'VMR'),
                'flow_data': 'VMR_VTU'
            },
            'field_statistics': {}
        }

        # 计算字段统计
        for field_name, field_data in result['fields'].items():
            if field_name == 'velocity':
                # 速度场是矢量场
                speed = np.sqrt(np.sum(field_data**2, axis=-1))
                metadata['field_statistics'][field_name] = {
                    'min': float(np.min(speed)),
                    'max': float(np.max(speed)),
                    'mean': float(np.mean(speed)),
                    'non_zero_count': int(np.count_nonzero(speed))
                }
            else:
                # 标量场
                metadata['field_statistics'][field_name] = {
                    'min': float(np.min(field_data)),
                    'max': float(np.max(field_data)),
                    'mean': float(np.mean(field_data)),
                    'non_zero_count': int(np.count_nonzero(field_data))
                }

        success = writer.save(result, hdf5_filename, metadata)
        hdf5_time = time.time() - start_time

        if success:
            print(f"  [OK] HDF5保存完成 ({hdf5_time:.1f}s)")
            print(f"    - 文件: {hdf5_filename}")
            print(f"    - 文件大小: {os.path.getsize(hdf5_filename)/(1024*1024):.2f} MB")
        else:
            print(f"  [ERROR] HDF5保存失败")
            return False

        # 7. 创建VTS可视化文件
        print(f"\n6. 创建VTS可视化文件")
        start_time = time.time()

        vts_filename = hdf5_filename.replace('.h5', '.vts')
        writer.convert_to_vtk(hdf5_filename, vts_filename)
        success = True  # convert_to_vtk doesn't return success status

        vts_time = time.time() - start_time

        if success:
            print(f"  [OK] VTS文件创建完成 ({vts_time:.1f}s)")
            print(f"    - 文件: {vts_filename}")
            print(f"    - 文件大小: {os.path.getsize(vts_filename)/(1024*1024):.2f} MB")
        else:
            print(f"  [ERROR] VTS文件创建失败")

        # 8. 性能总结
        total_time = data_prep_time + interpolation_time + hdf5_time + vts_time
        print(f"\n=== 性能总结 ===")
        print(f"数据准备: {data_prep_time:.1f}s")
        print(f"插值计算: {interpolation_time:.1f}s")
        print(f"HDF5保存: {hdf5_time:.1f}s")
        print(f"VTS创建:  {vts_time:.1f}s")
        print(f"总计:     {total_time:.1f}s")
        print(f"网格规模: {grid_size[0]*grid_size[1]*grid_size[2]:,}点")

        # 9. 验证结果
        print(f"\n=== 结果验证 ===")
        print(f"输出文件:")
        print(f"  - HDF5: {os.path.exists(hdf5_filename)}")
        print(f"  - VTS:  {os.path.exists(vts_filename)}")

        if 'pressure' in result['fields']:
            pressure = result['fields']['pressure']
            print(f"压力场验证:")
            print(f"  - 范围: [{np.min(pressure):.6f}, {np.max(pressure):.6f}]")
            print(f"  - 血管内非零值: {np.sum(pressure != 0):,}")

        if 'velocity' in result['fields']:
            velocity = result['fields']['velocity']
            speed = np.sqrt(np.sum(velocity**2, axis=-1))
            print(f"速度场验证:")
            print(f"  - 速度范围: [{np.min(speed):.6f}, {np.max(speed):.6f}]")
            print(f"  - 血管内非零值: {np.sum(speed != 0):,}")

        print(f"\n[OK] VMR完整工作流程测试成功！")
        return True

    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_vmr_complete_workflow()
    if success:
        print("\n🎉 VMR数据处理端到端验证完成！")
        print("VMR数据结构已完全集成，可以进行批量处理")
    else:
        print("\n❌ VMR工作流程需要进一步调试")