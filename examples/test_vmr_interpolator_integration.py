"""
测试VMR数据与插值器的集成
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vmr_data_loader import VMRDataLoader
from interpolator import GridInterpolator
import numpy as np


def test_vmr_interpolator_integration():
    """测试VMR数据加载器与插值器的完整集成"""
    print("=== VMR-插值器集成测试 ===")

    try:
        # 1. 创建VMR数据加载器
        loader = VMRDataLoader()
        case_name = "0007_H_AO_H"

        print(f"\n1. 加载VMR算例: {case_name}")
        interpolator_data = loader.create_interpolator_data(
            case_name=case_name,
            grid_size=(32, 32, 32),  # 使用小网格进行快速测试
            fields=['pressure', 'velocity']
        )

        if interpolator_data is None:
            print(f"  [ERROR] VMR数据加载失败")
            return False

        print(f"  [OK] VMR数据加载成功")
        print(f"    - 顶点数: {interpolator_data['num_points']:,}")
        print(f"    - 单元数: {interpolator_data['num_cells']:,}")
        print(f"    - 物理场: {list(interpolator_data['point_data'].keys())}")

        # 2. 创建插值器
        print(f"\n2. 创建插值器")
        interpolator = GridInterpolator(
            grid_size=(32, 32, 32),
            use_sdf=True
        )

        print(f"  [OK] 插值器创建成功")

        # 3. 准备插值器数据格式
        # VMR数据加载器已经提供了兼容的格式，但需要确保是interpolator期望的格式
        vtk_compatible_data = {
            'type': 'UnstructuredGrid',
            'blocks': [interpolator_data]  # 插值器期望blocks格式
        }

        print(f"\n3. 执行插值")
        result = interpolator.interpolate(vtk_compatible_data, fields=['pressure', 'velocity'])

        if result is None:
            print(f"  [ERROR] 插值失败")
            return False

        print(f"  [OK] 插值成功")
        print(f"    - 网格尺寸: {result['grid_size']}")
        print(f"    - 边界范围: {result['bounds']}")
        print(f"    - SDF使用: {result['sdf_used']}")
        print(f"    - 内部点: {result['inside_point_count']:,}")
        print(f"    - 外部点: {result['outside_point_count']:,}")
        print(f"    - 插值字段: {list(result['fields'].keys())}")

        # 4. 验证插值结果
        print(f"\n4. 验证插值结果")

        # 检查压力场
        if 'pressure' in result['fields']:
            pressure = result['fields']['pressure']
            print(f"  - 压力场:")
            print(f"    * 形状: {pressure.shape}")
            print(f"    * 范围: [{np.min(pressure):.6f}, {np.max(pressure):.6f}]")
            print(f"    * 非零值: {np.count_nonzero(pressure):,}")

        # 检查速度场
        if 'velocity' in result['fields']:
            velocity = result['fields']['velocity']
            print(f"  - 速度场:")
            print(f"    * 形状: {velocity.shape}")
            print(f"    * X范围: [{np.min(velocity[...,0]):.6f}, {np.max(velocity[...,0]):.6f}]")
            print(f"    * Y范围: [{np.min(velocity[...,1]):.6f}, {np.max(velocity[...,1]):.6f}]")
            print(f"    * Z范围: [{np.min(velocity[...,2]):.6f}, {np.max(velocity[...,2]):.6f}]")

        # 检查SDF场
        if 'SDF' in result['fields']:
            sdf = result['fields']['SDF']
            print(f"  - SDF场:")
            print(f"    * 形状: {sdf.shape}")
            print(f"    * 范围: [{np.min(sdf):.6f}, {np.max(sdf):.6f}]")
            positive_count = np.sum(sdf > 0)
            negative_count = np.sum(sdf < 0)
            print(f"    * 正值(内部): {positive_count:,}")
            print(f"    * 负值(外部): {negative_count:,}")

        print(f"\n[OK] VMR-插值器集成测试成功！")
        return True

    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_vmr_interpolator_integration()
    if success:
        print("\n[OK] VMR数据与插值器完全兼容！")
    else:
        print("\n[ERROR] VMR数据与插值器集成需要进一步调试")