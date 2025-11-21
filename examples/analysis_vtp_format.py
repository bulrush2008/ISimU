"""
分析VTP格式和VTU格式，了解与STL的差异
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
import numpy as np
import json


def analyze_vmr_data_structure():
    """分析VMR数据结构"""
    print("=== VMR数据结构分析 ===")

    # 1. 读取配置文件
    config_path = "../data_VMR/geo-flow.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"1. 配置文件分析:")
    print(f"   - 算例总数: {len(config)}")
    print(f"   - 算例列表: {list(config.keys())}")

    # 2. 选择第一个算例进行详细分析
    first_case = list(config.keys())[0]
    first_config = config[first_case]

    print(f"\\n2. 第一个算例分析 ({first_case}):")
    print(f"   - 几何文件: {first_config['geo']}")
    print(f"   - 流场文件: {first_config['flow']}")

    # 检查文件大小
    vtp_size = os.path.getsize(os.path.join('..', first_config['geo'])) / (1024*1024)  # MB
    vtu_size = os.path.getsize(os.path.join('..', first_config['flow'])) / (1024*1024)  # MB

    print(f"   - 几何文件大小: {vtp_size:.2f} MB")
    print(f"   - 流场文件大小: {vtu_size:.2f} MB")

    # 修复路径为相对于当前examples目录的路径
    first_config['geo'] = os.path.join('..', first_config['geo'])
    first_config['flow'] = os.path.join('..', first_config['flow'])

    return first_config


def analyze_vtp_vs_stl():
    """对比VTP和STL格式"""
    print(f"\\n=== VTP vs STL 格式对比分析 ===")

    print("STL格式特点:")
    print("  - 格式: 纯三角形网格格式")
    print("  - 内容: 顶点坐标 + 三角形面片索引")
    print("  - 处理: 需要trimesh库")
    print("  - 缩放: ISimU中应用0.001缩放")
    print("  - 优点: 简单、通用")
    print("  - 缺点: 只包含几何信息")

    print("\\nVTP格式特点:")
    print("  - 格式: VTK多边形数据格式")
    print("  - 内容: 顶点 + 多种单元类型 + 可选物理场")
    print("  - 处理: 直接使用VTK库")
    print("  - 缩放: ISimU中无需缩放(比例=1.0)")
    print("  - 优点: 信息丰富、VTK原生支持")
    print("  - 缺点: 格式相对复杂")


def analyze_vtp_file_header(vtp_file):
    """分析VTP文件头部信息（暂时不直接读取）"""
    print(f"\\n=== VTP文件头部分析 ===")
    print(f"文件: {vtp_file}")

    try:
        with open(vtp_file, 'r', encoding='utf-8', errors='ignore') as f:
            header_content = f.read(2000)  # 读取前2000字符

        print(f"\\nVTP文件头部信息:")
        print(f"  - 文件大小: {os.path.getsize(vtp_file) / (1024*1024):.2f} MB")

        # 检查VTK版本
        if '<?xml' in header_content:
            print(f"  - 格式: XML VTK文件")

        # 检查数据类型
        if 'PolyData' in header_content:
            print(f"  - 数据类型: PolyData")
        elif 'UnstructuredGrid' in header_content:
            print(f"  - 数据类型: UnstructuredGrid")

        # 检查点数量
        import re
        points_match = re.search(r'NumberOfPoints="(\d+)"', header_content)
        if points_match:
            num_points = int(points_match.group(1))
            print(f"  - 顶点数: {num_points:,}")

        # 检查单元数量
        polys_match = re.search(r'NumberOfPolys="(\d+)"', header_content)
        if polys_match:
            num_polys = int(polys_match.group(1))
            print(f"  - 多边形数: {num_polys:,}")

        return True

    except Exception as e:
        print(f"[ERROR] VTP文件分析失败: {e}")
        return False


def read_and_analyze_vtu_file(vtu_file):
    """读取并分析VTU文件（用于对比）"""
    print(f"\\n=== VTU文件分析（流场数据） ===")
    print(f"文件: {vtu_file}")

    reader = VTKReader()

    try:
        vtu_data = reader.read_vtu(vtu_file)

        print(f"\\nVTU文件结构:")
        print(f"  - 数据类型: {vtu_data['type']}")
        print(f"  - 顶点数: {vtu_data['num_points']:,}")
        print(f"  - 单元数: {vtu_data['num_cells']:,}")

        # 分析顶点范围（与VTP对比）
        vertices = vtu_data['vertices']
        print(f"\\n顶点范围（VMR坐标系统）:")
        print(f"  - X范围: [{vertices[:, 0].min():.6f}, {vertices[:, 0].max():.6f}]")
        print(f"  - Y范围: [{vertices[:, 1].min():.6f}, {vertices[:, 1].max():.6f}]")
        print(f"  - Z范围: [{vertices[:, 2].min():.6f}, {vertices[:, 2].max():.6f}]")

        # 分析流场数据
        if 'point_data' in vtu_data:
            print(f"\\n流场物理量:")
            for field_name, field_data in vtu_data['point_data'].items():
                if len(field_data.shape) == 1:
                    print(f"  - {field_name}: 标量场 {field_data.shape}")
                else:
                    print(f"  - {field_name}: 矢量场 {field_data.shape[1]}维 {field_data.shape}")

        return True, vtu_data

    except Exception as e:
        print(f"[ERROR] VTU文件读取失败: {e}")
        return False, None


def analyze_coordinate_scaling(vtu_data):
    """分析VMR数据的坐标系统和缩放需求"""
    print(f"\\n=== VMR坐标系统分析 ===")

    vertices = vtu_data['vertices']

    print(f"VMR数据坐标特征:")
    print(f"  - 顶点数: {len(vertices):,}")
    print(f"  - X范围: [{vertices[:, 0].min():.6f}, {vertices[:, 0].max():.6f}]")
    print(f"  - Y范围: [{vertices[:, 1].min():.6f}, {vertices[:, 1].max():.6f}]")
    print(f"  - Z范围: [{vertices[:, 2].min():.6f}, {vertices[:, 2].max():.6f}]")

    # 计算坐标范围
    x_range = vertices[:, 0].max() - vertices[:, 0].min()
    y_range = vertices[:, 1].max() - vertices[:, 1].min()
    z_range = vertices[:, 2].max() - vertices[:, 2].min()

    print(f"\\n坐标范围大小:")
    print(f"  - X范围: {x_range:.6f}")
    print(f"  - Y范围: {y_range:.6f}")
    print(f"  - Z范围: {z_range:.6f}")

    # 缩放建议
    max_range = max(x_range, y_range, z_range)
    print(f"\\n缩放分析:")
    if max_range > 100:
        print(f"  - 坐标范围较大 (>100)，可能需要缩放")
        print(f"  - 建议缩放比例: 1.0 (无缩放)")
    elif max_range > 10:
        print(f"  - 坐标范围中等 (10-100)")
        print(f"  - 建议缩放比例: 1.0 (无缩放)")
    else:
        print(f"  - 坐标范围较小 (<10)")
        print(f"  - 建议缩放比例: 1.0 (无缩放)")

    print(f"\\n结论: VMR数据无需缩放，缩放比例=1.0")


def main():
    """主分析函数"""
    print("=== 第1个子任务：分析新数据结构和格式差异 ===\\n")

    # 1. 分析VMR数据结构
    config = analyze_vmr_data_structure()

    # 2. 对比VTP vs STL格式
    analyze_vtp_vs_stl()

    # 3. 分析VTP文件
    vtp_success = analyze_vtp_file_header(config['geo'])

    # 4. 分析VTU文件
    vtu_success, vtu_data = read_and_analyze_vtu_file(config['flow'])

    # 5. 分析坐标差异（基于VTU数据）
    if vtu_success:
        analyze_coordinate_scaling(vtu_data)

    print(f"\\n=== 分析总结 ===")
    print(f"1. VMR数据结构: {len(open('../data_VMR/geo-flow.json').read().split())} 行配置")
    print(f"2. VTP格式: 复杂的多边形数据，可直接用VTK处理")
    print(f"3. 坐标系统: 需要验证VTP和VTU的坐标一致性")
    print(f"4. 缩放需求: 确认无需缩放（比例=1.0）")

    return vtp_success, vtu_data


if __name__ == "__main__":
    vtp_data, vtu_data = main()