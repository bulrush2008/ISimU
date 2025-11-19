"""
完整的数据处理流程示例

VTK读取 -> 网格插值 -> HDF5存储
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
from interpolator import GridInterpolator
from hdf5_storage import HDF5Storage
import numpy as np


def main():
    """完整的数据处理流程"""
    print("=== ISimU Complete Data Processing Pipeline ===\n")

    # 配置参数
    vtm_file = "../Data/vessel.000170.vtm"
    output_h5 = "../matrix_data/output_vessel_170.h5"
    output_vtk = "../matrix_data/output_vessel_170.vts"

    # 插值参数
    grid_size = (32, 32, 32)  # 可以调整这个值来改变输出网格密度
    interpolation_method = 'linear'  # 'linear', 'nearest', 'cubic'

    print(f"配置参数:")
    print(f"  - 输入文件: {vtm_file}")
    print(f"  - 网格尺寸: {grid_size}")
    print(f"  - 插值方法: {interpolation_method}")
    print(f"  - 输出HDF5: {output_h5}")
    print(f"  - 输出VTK: {output_vtk}\n")

    try:
        # 第一步：读取VTK文件
        print("第1步: 读取VTK文件...")
        reader = VTKReader()
        vtk_data = reader.read_vtm(vtm_file)

        # 获取可用场变量
        available_fields = reader.get_available_fields(vtk_data)
        print(f"  ✓ 成功读取VTK文件")
        print(f"  - 数据块数量: {vtk_data['num_blocks']}")
        print(f"  - 可用场变量: {available_fields}")

        # 第二步：网格插值
        print(f"\n第2步: 执行网格插值...")
        interpolator = GridInterpolator(
            grid_size=grid_size,
            method=interpolation_method
        )

        # 选择要插值的场变量（可以选择所有或特定变量）
        fields_to_interpolate = available_fields[:3] if len(available_fields) > 3 else available_fields
        print(f"  - 插值场变量: {fields_to_interpolate}")

        interpolated_data = interpolator.interpolate(vtk_data, fields_to_interpolate)

        # 获取插值统计信息
        stats = interpolator.get_interpolation_statistics(interpolated_data)
        print(f"  ✓ 插值完成")
        print(f"  - 总网格点数: {stats['total_points']:,}")

        for field_name, field_stats in stats['field_statistics'].items():
            print(f"    {field_name}: 范围[{field_stats['min']:.3e}, {field_stats['max']:.3e}], "
                  f"均值{field_stats['mean']:.3e}")

        # 第三步：保存为HDF5格式
        print(f"\n第3步: 保存为HDF5格式...")
        storage = HDF5Storage()

        # 准备元数据
        metadata = {
            'source_file': vtm_file,
            'grid_size': grid_size,
            'interpolation_method': interpolation_method,
            'original_fields': available_fields,
            'interpolated_fields': fields_to_interpolate,
            'description': 'ISimU插值数据 - vessel算例时间步170'
        }

        storage.save(interpolated_data, output_h5, metadata)
        print(f"  ✓ HDF5文件保存完成")

        # 验证保存的文件
        file_info = storage.get_file_info(output_h5)
        print(f"  - 文件大小: {file_info['total_data_size_mb']:.2f} MB")

        # 第四步：转换为VTK格式用于可视化
        print(f"\n第4步: 转换为VTK格式...")
        try:
            storage.convert_to_vtk(output_h5, output_vtk)
            print(f"  ✓ VTK文件保存完成")
            print(f"  - 可以用ParaView打开: {output_vtk}")
        except Exception as e:
            print(f"  VTK转换失败: {e}")

        print(f"\n=== 处理完成 ===")
        print(f"输出文件:")
        print(f"  - HDF5数据: {output_h5}")
        print(f"  - VTK可视化: {output_vtk}")

        return True

    except Exception as e:
        print(f"\n✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_validation():
    """测试数据验证功能"""
    print("\n=== 数据验证测试 ===")

    try:
        from hdf5_storage import HDF5Storage

        # 加载之前生成的HDF5文件
        h5_file = "../matrix_data/output_vessel_170.h5"
        if not os.path.exists(h5_file):
            print("  跳过验证：HDF5文件不存在")
            return

        storage = HDF5Storage()
        data = storage.load(h5_file)

        print(f"  ✓ 成功加载HDF5文件")
        print(f"  - 网格尺寸: {data['grid_size']}")
        print(f"  - 边界范围: {data['bounds']}")

        # 验证数据完整性
        for field_name, field_data in data['fields'].items():
            nan_count = np.sum(np.isnan(field_data))
            inf_count = np.sum(np.isinf(field_data))
            zero_count = np.sum(field_data == 0)

            print(f"  - {field_name}:")
            print(f"    形状: {field_data.shape}")
            print(f"    NaN点: {nan_count}")
            print(f"    无穷值点: {inf_count}")
            print(f"    零值点: {zero_count}")

            if nan_count > 0 or inf_count > 0:
                print(f"    ⚠️  数据质量问题检测到")
            else:
                print(f"    ✓ 数据质量良好")

    except Exception as e:
        print(f"  ✗ 验证失败: {e}")


if __name__ == "__main__":
    success = main()

    if success:
        test_data_validation()

    if not success:
        sys.exit(1)