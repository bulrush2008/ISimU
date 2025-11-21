"""
测试VTP格式几何文件的SDF计算
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from geometry_reader import GeometryReader
from sdf_utils_enhanced import EnhancedSDFCalculator
import numpy as np


def test_vtp_geometry_reading():
    """测试VTP几何文件读取"""
    print("=== VTP几何文件读取测试 ===")

    reader = GeometryReader()

    # 测试VTP文件
    vtp_file = "../data_VMR/0007_H_AO_H/Simulations/0090_0001/check/initial.vtp"

    if not os.path.exists(vtp_file):
        print(f"[ERROR] VTP文件不存在: {vtp_file}")
        return False

    print(f"读取VTP文件: {vtp_file}")

    try:
        geometry = reader.read_geometry(vtp_file)

        if geometry:
            print(f"[OK] VTP几何读取成功")
            print(f"  - 格式: {geometry['format']}")
            print(f"  - 顶点数: {geometry['num_vertices']:,}")
            print(f"  - 面片数: {geometry['num_faces']:,}")
            print(f"  - 缩放比例: {geometry['scale_factor']}")
            print(f"  - 文件路径: {geometry['file_path']}")

            # 检查顶点范围
            vertices = geometry['vertices']
            print(f"  - X范围: [{vertices[:, 0].min():.6f}, {vertices[:, 0].max():.6f}]")
            print(f"  - Y范围: [{vertices[:, 1].min():.6f}, {vertices[:, 1].max():.6f}]")
            print(f"  - Z范围: [{vertices[:, 2].min():.6f}, {vertices[:, 2].max():.6f}]")

            # 检查面片结构
            faces = geometry['faces']
            if faces is not None:
                print(f"  - 面片形状: {faces.shape}")
                unique_sizes = np.unique(faces.shape[1] if len(faces.shape) > 1 else 1)
                print(f"  - 面片类型: {unique_sizes}")

                # 统计不同类型的面片
                face_types = {}
                for i in range(faces.shape[0]):
                    n_verts = len(faces[i]) if len(faces.shape) > 1 else 1
                    if n_verts not in face_types:
                        face_types[n_verts] = 0
                    face_types[n_verts] += 1

                for n_verts, count in face_types.items():
                    print(f"    - {n_verts}边形: {count:,}个")

            return geometry

        else:
            print(f"[ERROR] VTP几何读取失败")
            return False

    except Exception as e:
        print(f"[ERROR] 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vtp_sdf_creation():
    """测试从VTP创建SDF"""
    print("\n=== VTP SDF创建测试 ===")

    calculator = EnhancedSDFCalculator()

    vtp_file = "../data_VMR/0007_H_AO_H/Simulations/0090_0001/check/initial.vtp"

    if not os.path.exists(vtp_file):
        print(f"[ERROR] VTP文件不存在: {vtp_file}")
        return False

    print(f"从VTP创建SDF: {vtp_file}")

    try:
        sdf = calculator.create_sdf_from_file(vtp_file)

        if sdf:
            print(f"[OK] VTP SDF创建成功")

            # 获取几何信息
            info = calculator.get_current_geometry_info()
            print(f"  - 几何格式: {info['format']}")
            print(f"  - 缩放比例: {info['scale_factor']}")
            print(f"  - 顶点数: {info['num_vertices']:,}")
            print(f"  - 面片数: {info['num_faces']:,}")

            # 验证SDF
            is_valid, issues = calculator.validate_sdf()
            print(f"  - SDF验证: {'通过' if is_valid else '失败'}")
            if issues:
                for issue in issues:
                    print(f"      * {issue}")

            # 测试SDF计算
            print(f"  - 测试SDF计算...")

            # 创建测试点（包括几何内部、外部和边界点）
            test_points = [
                [-5.0, -3.0, -10.0],  # 最小边界点
                [0.0, 0.0, 0.0],  # 原点
                [5.0, 2.0, 5.0],  # 中间点
                [10.0, 10.0, 20.0]  # 远离点
            ]

            sdf_values = sdf.compute_sdf(np.array(test_points))
            print(f"    测试点SDF值:")
            for i, (point, value) in enumerate(zip(test_points, sdf_values)):
                print(f"      点{i+1} {point}: {value:.6f}")

            # 分析SDF值分布
            positive_count = np.sum(sdf_values > 0)
            negative_count = np.sum(sdf_values < 0)
            zero_count = np.sum(sdf_values == 0)

            print(f"    SDF值分布:")
            print(f"      正值(内部): {positive_count}")
            print(f"      负值(外部): {negative_count}")
            print(f"      零值(表面): {zero_count}")

            return sdf

        else:
            print(f"[ERROR] VTP SDF创建失败")
            return False

    except Exception as e:
        print(f"[ERROR] SDF创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vmr_config_sdf():
    """测试从VMR配置创建SDF"""
    print("\n=== VMR配置SDF测试 ===")

    calculator = EnhancedSDFCalculator()

    config_file = "../data_VMR/geo-flow.json"

    if not os.path.exists(config_file):
        print(f"[ERROR] VMR配置文件不存在: {config_file}")
        return False

    print(f"从VMR配置创建SDF: 0007_H_AO_H")

    try:
        sdf = calculator.create_sdf_from_vmr_config("0007_H_AO_H")

        if sdf:
            print(f"[OK] VMR SDF创建成功")

            info = calculator.get_current_geometry_info()
            print(f"  - 几何源: {info['file_path']}")
            print(f"  - 格式: {info['format']}")
            print(f"  - 缩放: {info['scale_factor']}")
            print(f"  - 顶点数: {info['num_vertices']:,}")

            return sdf

        else:
            print(f"[ERROR] VMR SDF创建失败")
            return False

    except Exception as e:
        print(f"[ERROR] VMR SDF测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=== 第3个子任务：实现VTP几何读取器 ===\n")

    # 测试1: VTP几何读取
    geometry = test_vtp_geometry_reading()

    # 测试2: VTP SDF创建
    sdf = test_vtp_sdf_creation()

    # 测试3: VMR配置SDF创建
    vmr_sdf = test_vmr_config_sdf()

    print(f"\n=== 测试总结 ===")

    success_count = 0
    total_tests = 3

    if geometry:
        print(f"[OK] VTP几何读取: 成功")
        success_count += 1
    else:
        print(f"[ERROR] VTP几何读取: 失败")

    if sdf:
        print(f"[OK] VTP SDF创建: 成功")
        success_count += 1
    else:
        print(f"[ERROR] VTP SDF创建: 失败")

    if vmr_sdf:
        print(f"[OK] VMR配置SDF: 成功")
        success_count += 1
    else:
        print(f"[ERROR] VMR配置SDF: 失败")

    print(f"\n总体结果: {success_count}/{total_tests} 测试通过")

    if success_count == total_tests:
        print("🎉 VTP几何读取器实现完成！")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)