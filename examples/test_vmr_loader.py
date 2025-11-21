"""
测试VMR数据加载器的简化版本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_vmr_simple():
    """简化的VMR加载器测试"""
    print("=== VMR数据加载器测试 ===")

    try:
        # 直接使用配置文件中的路径
        config_file = "../data_VMR/geo-flow.json"

        if not os.path.exists(config_file):
            print(f"[ERROR] 配置文件不存在: {config_file}")
            return False

        import json
        with open(config_file, 'r') as f:
            config = json.load(f)

        print(f"配置文件加载成功: {config_file}")
        print(f"可用算例: {list(config.keys())}")

        # 选择第一个算例
        first_case = list(config.keys())[0]
        case_config = config[first_case]

        # 从examples目录访问，需要正确处理路径
        geometry_path = "../" + case_config['geo']
        flow_path = "../" + case_config['flow']

        print(f"\\n测试算例: {first_case}")
        print(f"  - 几何文件: {geometry_path}")
        print(f"  - 流场文件: {flow_path}")
        print(f"  - 几何存在: {os.path.exists(geometry_path)}")
        print(f"  - 流场存在: {os.path.exists(flow_path)}")

        if os.path.exists(geometry_path) and os.path.exists(flow_path):
            print(f"  [OK] 文件存在，测试数据加载...")

            # 测试几何读取
            from geometry_reader import GeometryReader
            reader = GeometryReader()

            geometry_data = reader.read_geometry(geometry_path)
            if geometry_data:
                print(f"  [OK] 几何数据读取成功: {geometry_data['num_vertices']:,}顶点")
            else:
                print(f"  [ERROR] 几何数据读取失败")
                return False

            # 测试流场读取
            from data_reader import VTKReader
            vtk_reader = VTKReader()

            flow_data = vtk_reader.read_vtu(flow_path)
            if flow_data:
                print(f"  [OK] 流场数据读取成功: {flow_data['num_points']:,}点")
            else:
                print(f"  [ERROR] 流场数据读取失败")
                return False

            # 测试SDF创建
            from sdf_utils_enhanced import EnhancedSDFCalculator
            calculator = EnhancedSDFCalculator()

            sdf = calculator.create_sdf_from_data(geometry_data)
            if sdf:
                print(f"  [OK] SDF计算器创建成功")
            else:
                print(f"  [ERROR] SDF计算器创建失败")
                return False

            print(f"\\n[OK] VMR数据加载器完全正常工作！")
            return True
        else:
            print(f"[ERROR] 文件不存在，无法继续测试")
            return False

    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_vmr_simple()
    if success:
        print("\n[OK] VMR数据加载器实现完成！")
    else:
        print("\n[ERROR] VMR数据加载器需要进一步调试")