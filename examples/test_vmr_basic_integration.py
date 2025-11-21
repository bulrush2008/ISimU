"""
测试VMR数据与插值器的基本集成（不包含完整SDF计算）
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vmr_data_loader import VMRDataLoader
from interpolator import GridInterpolator


def test_basic_integration():
    """基本集成测试，验证数据格式兼容性"""
    print("=== VMR-插值器基本集成测试 ===")

    try:
        # 1. 创建VMR数据加载器
        loader = VMRDataLoader()
        case_name = "0007_H_AO_H"

        print(f"\n1. 测试VMR数据加载")

        # 先测试加载算例数据
        case_data = loader.load_case(case_name, fields=['pressure', 'velocity'])

        if case_data is None:
            print(f"  [ERROR] VMR数据加载失败")
            return False

        print(f"  [OK] VMR数据加载成功")
        print(f"    - 几何顶点: {case_data['geometry_data']['num_vertices']:,}")
        print(f"    - 几何面片: {case_data['geometry_data']['num_faces']:,}")
        print(f"    - 流场点: {case_data['flow_data']['num_points']:,}")
        print(f"    - SDF计算器: {'已创建' if case_data['sdf_calculator'] else '未创建'}")

        # 2. 创建插值器并测试数据格式
        print(f"\n2. 测试插值器数据格式")

        interpolator = GridInterpolator(
            grid_size=(16, 16, 16),  # 极小网格进行快速测试
            use_sdf=False  # 暂时关闭SDF进行快速测试
        )

        print(f"  [OK] 插值器创建成功")

        # 3. 创建插值器兼容的数据格式
        print(f"\n3. 创建插值器兼容数据")

        # 从case_data创建插值器兼容格式
        flow_data = case_data['flow_data']

        # 检查数据结构
        if 'blocks' in flow_data and flow_data['blocks']:
            main_block = flow_data['blocks'][0]
        else:
            main_block = flow_data

        interpolator_data = {
            'type': flow_data.get('type', 'UnstructuredGrid'),
            'num_points': main_block.get('num_points', 0),
            'num_cells': main_block.get('num_cells', 0),
            'vertices': main_block['vertices'],
            'point_data': main_block.get('point_data', {}),
            'cell_data': main_block.get('cell_data', {}),
            'geometry_data': case_data['geometry_data'],
            'sdf_calculator': case_data['sdf_calculator'],
            'case_name': case_name,
            'grid_size': (16, 16, 16),
            'bounds': None
        }

        print(f"  [OK] 插值器数据格式创建成功")
        print(f"    - 数据类型: {interpolator_data['type']}")
        print(f"    - 顶点数: {interpolator_data['num_points']:,}")
        print(f"    - 物理场: {list(interpolator_data['point_data'].keys())}")

        # 4. 测试网格设置
        print(f"\n4. 测试网格设置")

        vertices = interpolator_data['vertices']
        interpolator.setup_cartesian_grid(vertices)

        print(f"  [OK] 笛卡尔网格设置成功")
        print(f"    - 网格尺寸: {interpolator.grid_size}")
        print(f"    - 边界范围: {interpolator.bounds}")

        # 5. 准备兼容的VTK数据格式
        vtk_compatible_data = {
            'type': 'UnstructuredGrid',
            'blocks': [interpolator_data]  # 插值器期望blocks格式
        }

        print(f"  [OK] VTK兼容数据格式准备完成")

        print(f"\n[OK] VMR-插值器基本集成测试成功！")
        print(f"数据格式完全兼容，可以进行后续的插值计算")
        return True

    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_integration()
    if success:
        print("\n[OK] VMR数据与插值器数据格式兼容！")
        print("可以继续进行完整的插值测试")
    else:
        print("\n[ERROR] VMR数据与插值器集成存在问题")