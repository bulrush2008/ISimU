"""
ISimU 可配置网格插值应用
基于test_32x32x32_quick_verification.py，支持用户自定义网格大小
"""

import sys
import os
import argparse
import numpy as np
from datetime import datetime

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
from interpolator_optimized import OptimizedGridInterpolator
from hdf5_storage import HDF5Storage


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='ISimU网格插值应用 - 支持自定义网格大小',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 默认32x32x32网格
  python interpolation_app.py

  # 自定义64x64x64网格
  python interpolation_app.py --grid-size 64 64 64

  # 使用不同的批处理大小
  python interpolation_app.py --grid-size 48 48 48 --batch-size 15000

  # 指定输出文件名
  python interpolation_app.py --grid-size 32 32 32 --output-prefix my_interpolation
        """
    )

    parser.add_argument(
        '--grid-size', '-g',
        type=int,
        nargs=3,
        default=[32, 32, 32],
        metavar=('NX', 'NY', 'NZ'),
        help='网格尺寸 (默认: 32 32 32)'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=10000,
        help='SDF计算的批处理大小 (默认: 10000)'
    )

    parser.add_argument(
        '--method', '-m',
        choices=['linear', 'nearest'],
        default='linear',
        help='插值方法 (默认: linear)'
    )

    parser.add_argument(
        '--output-prefix', '-o',
        type=str,
        default='dense_interpolation',
        help='输出文件名前缀 (默认: dense_interpolation)'
    )

    parser.add_argument(
        '--no-sdf',
        action='store_true',
        help='禁用SDF计算（对所有点进行插值）'
    )

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    grid_size = tuple(args.grid_size)

    print("=== ISimU 网格插值应用 ===")
    print(f"网格尺寸: {grid_size[0]}×{grid_size[1]}×{grid_size[2]} ({np.prod(grid_size):,} 个点)")
    print(f"插值方法: {args.method}")
    print(f"批处理大小: {args.batch_size}")
    print(f"SDF计算: {'禁用' if args.no_sdf else '启用'}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 验证网格大小合理性
    total_points = np.prod(grid_size)
    if total_points > 10000000:  # 10M点警告
        print(f"[WARNING] 网格规模较大 ({total_points:,} 点)，处理时间可能很长")
        response = input("是否继续? (y/N): ").strip().lower()
        if response != 'y':
            print("已取消")
            return False
    print()

    # 初始化组件
    reader = VTKReader()
    interpolator = OptimizedGridInterpolator(
        grid_size=grid_size,
        method=args.method,
        use_sdf=not args.no_sdf,
        batch_size=args.batch_size
    )
    storage = HDF5Storage()

    # 输入文件路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vtu_file = os.path.join(base_dir, "data_UNS", "vessel", "170", "Part.0.Zone.1.vtu")
    print(f"输入文件: {vtu_file}")

    # 验证文件存在
    if not os.path.exists(vtu_file):
        print(f"[ERROR] 输入文件不存在: {vtu_file}")
        return False

    # 输出文件路径
    output_dir = os.path.join(base_dir, "data_matrix")
    os.makedirs(output_dir, exist_ok=True)

    grid_str = f"{grid_size[0]}x{grid_size[1]}x{grid_size[2]}"
    output_h5 = os.path.join(output_dir, f"{args.output_prefix}_{grid_str}.h5")
    output_vts = os.path.join(output_dir, f"{args.output_prefix}_{grid_str}.vts")

    print(f"输出文件:")
    print(f"  - HDF5: {output_h5}")
    print(f"  - VTS:  {output_vts}")
    print()

    try:
        # 1. 读取VTU数据
        print("1. 读取VTU数据...")
        vtu_data = reader.read_vtu(vtu_file)
        print(f"  [OK] 成功读取VTU文件")
        print(f"  - 网格类型: {vtu_data['type']}")
        print(f"  - 点数: {vtu_data['num_points']:,}")
        print(f"  - 单元数: {vtu_data['num_cells']:,}")

        # 显示可用的物理场变量
        if 'point_data' in vtu_data:
            fields = list(vtu_data['point_data'].keys())
            print(f"  - 可用物理场: {fields}")
        else:
            print(f"  - [WARNING] 未找到点数据")
            return False
        print()

        # 2. 执行插值
        print(f"2. 执行{grid_str}网格插值...")
        start_time = datetime.now()

        interpolated_data = interpolator.interpolate(vtu_data, fields=fields)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"  [OK] 插值完成，耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
        print(f"  - 网格尺寸: {interpolated_data['grid_size']}")
        print(f"  - SDF使用: {interpolated_data['sdf_used']}")
        print()

        # 显示插值结果统计
        print("3. 插值结果统计:")
        for field_name, field_data in interpolated_data['fields'].items():
            field_info = {
                'shape': field_data.shape,
                'min': float(field_data.min()),
                'max': float(field_data.max()),
                'mean': float(field_data.mean()),
                'nan_count': int(np.sum(np.isnan(field_data))),
                'zero_count': int(np.sum(field_data == 0))
            }
            print(f"  - {field_name}:")
            print(f"    形状: {field_info['shape']}")
            print(f"    范围: [{field_info['min']:.3e}, {field_info['max']:.3e}]")
            print(f"    均值: {field_info['mean']:.3e}")
            print(f"    NaN点数: {field_info['nan_count']}")
            print(f"    零值点数: {field_info['zero_count']}")
        print()

        # 4. 保存HDF5文件
        print("4. 保存HDF5文件...")
        metadata = {
            'source_file': vtu_file,
            'grid_size': interpolated_data['grid_size'],
            'interpolation_method': args.method,
            'use_sdf': interpolated_data['sdf_used'],
            'batch_size': args.batch_size,
            'processing_time_seconds': duration,
            'total_points': int(total_points),
            'creation_time': datetime.now().isoformat(),
            'description': f'{grid_str}网格插值结果，ISimU可配置插值应用生成'
        }

        storage.save(interpolated_data, output_h5, metadata=metadata)
        print(f"  [OK] HDF5文件已保存")

        # 显示文件大小
        file_size = os.path.getsize(output_h5) / (1024 * 1024)  # MB
        print(f"  - 文件大小: {file_size:.2f} MB")
        print()

        # 5. 转换为VTK格式
        print("5. 转换为VTK格式...")
        try:
            storage.convert_to_vtk(output_h5, output_vts)
            print(f"  [OK] VTK文件已保存")

            # 显示VTS文件大小
            vts_size = os.path.getsize(output_vts) / (1024 * 1024)  # MB
            print(f"  - 文件大小: {vts_size:.2f} MB")
        except Exception as e:
            print(f"  [WARNING] VTK转换失败: {e}")
        print()

        # 6. 性能统计
        print("6. 性能统计:")
        points_per_second = total_points / duration
        print(f"  - 处理速度: {points_per_second:.0f} 点/秒")
        print(f"  - 内存效率: {file_size/total_points*1024:.2f} KB/千点")
        print()

        # 完成信息
        print("=== 插值完成 ===")
        print(f"网格尺寸: {grid_str}")
        print(f"处理时间: {duration:.1f}秒")
        print(f"输出文件:")
        print(f"  - HDF5: {output_h5}")
        print(f"  - VTS:  {output_vts}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return True

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断操作")
        return False
    except Exception as e:
        print(f"\n[ERROR] 插值失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)