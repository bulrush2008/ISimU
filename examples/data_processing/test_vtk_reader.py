"""
测试VTK文件读取功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader


def main():
    """主测试函数"""
    print("=== VTK文件读取测试 ===")

    reader = VTKReader()

    # 测试读取主VTM文件
    vtm_file = "../Data/vessel.000170.vtm"
    print(f"\n1. 读取VTM文件: {vtm_file}")

    try:
        data = reader.read_vtm(vtm_file)
        print(f"✓ 成功读取VTM文件")
        print(f"  - 数据块数量: {data['num_blocks']}")

        # 显示各数据块信息
        for i, block in enumerate(data['blocks']):
            print(f"  - 数据块 {i}: {block['type']}")
            print(f"    点数: {block['num_points']}")
            if 'num_cells' in block:
                print(f"    单元数: {block['num_cells']}")

            # 显示物理场变量
            if 'point_data' in block and block['point_data']:
                print(f"    点数据变量: {list(block['point_data'].keys())}")
            if 'cell_data' in block and block['cell_data']:
                print(f"    单元数据变量: {list(block['cell_data'].keys())}")

        # 显示所有可用场变量
        fields = reader.get_available_fields(data)
        print(f"\n✓ 所有可用物理场变量: {fields}")

    except Exception as e:
        print(f"✗ 读取失败: {e}")
        return False

    # 测试读取具体的VTU文件
    vtu_file = "../Data/vessel/170/Part.0.Zone.1.vtu"
    print(f"\n2. 读取VTU文件: {vtu_file}")

    try:
        vtu_data = reader.read_vtu(vtu_file)
        print(f"✓ 成功读取VTU文件")
        print(f"  - 网格类型: {vtu_data['type']}")
        print(f"  - 点数: {vtu_data['num_points']}")
        print(f"  - 单元数: {vtu_data['num_cells']}")

        # 显示数据形状
        if 'point_data' in vtu_data:
            for field_name, field_data in vtu_data['point_data'].items():
                print(f"  - {field_name}: {field_data.shape} ({field_data.dtype})")

    except Exception as e:
        print(f"✗ 读取失败: {e}")
        return False

    print(f"\n=== 测试完成 ===")
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)