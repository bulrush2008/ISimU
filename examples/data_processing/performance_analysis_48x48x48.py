"""
性能分析版本的48x48x48密集网格测试
详细分析每个步骤的执行时间，找出性能瓶颈
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
from interpolator import GridInterpolator
from hdf5_storage import HDF5Storage
from stl_reader import load_portal_vein_geometry
import numpy as np
import h5py


def time_function(func, name, *args, **kwargs):
    """测量函数执行时间的装饰器"""
    print(f"  开始执行: {name}")
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"  完成执行: {name} - 耗时: {elapsed:.2f}秒 ({elapsed/60:.1f}分钟)")
    return result, elapsed


def test_performance_analysis():
    """48x48x48网格性能分析测试"""
    print("=== 48x48x48 Dense Grid Performance Analysis ===\n")

    # 配置参数
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vtm_file = os.path.join(base_dir, "Data", "vessel.000170.vtm")
    grid_size = (48, 48, 48)

    print(f"配置信息:")
    print(f"  - 网格规模: {grid_size} ({np.prod(grid_size):,} 点)")
    print(f"  - VTM文件: {vtm_file}")
    print()

    # 记录总执行时间
    total_start = time.time()

    # 性能统计字典
    performance_stats = {}

    try:
        # 第一步：STL文件读取
        print("Step 1: STL文件读取")
        stl_data, stl_time = time_function(
            load_portal_vein_geometry, "STL几何文件读取", base_dir
        )
        performance_stats['stl_loading'] = stl_time
        print()

        # 第二步：VTK文件读取
        print("Step 2: VTK文件读取")
        reader = VTKReader()
        vtk_data, vtk_time = time_function(
            reader.read_vtm, "VTM文件读取", vtm_file
        )
        performance_stats['vtk_reading'] = vtk_time
        print()

        # 第三步：插值器初始化
        print("Step 3: 插值器初始化")
        def init_interpolator():
            return GridInterpolator(
                grid_size=grid_size,
                method='linear',
                out_of_domain_value=0.0,
                use_sdf=True
            )

        interpolator, init_time = time_function(
            init_interpolator, "GridInterpolator初始化"
        )
        performance_stats['interpolator_init'] = init_time
        print()

        # 第四步：SDF和插值计算（最耗时的部分）
        print("Step 4: SDF计算和插值")
        fields_to_interpolate = ['P', 'Velocity']

        def perform_sdf_and_interpolation():
            return interpolator.interpolate(vtk_data, fields_to_interpolate)

        result, sdf_interp_time = time_function(
            perform_sdf_and_interpolation, f"SDF计算+插值 ({', '.join(fields_to_interpolate)})"
        )
        performance_stats['sdf_and_interpolation'] = sdf_interp_time
        print()

        # 第五步：数据保存
        print("Step 5: 数据保存")

        # HDF5保存
        storage = HDF5Storage()
        output_h5 = os.path.join(base_dir, "matrix_data", "perf_analysis_48x48x48.h5")

        def save_hdf5():
            metadata = {
                'source_file': vtm_file,
                'grid_size': grid_size,
                'performance_analysis': True
            }
            storage.save(result, output_h5, metadata)

        _, save_time = time_function(
            save_hdf5, "HDF5文件保存"
        )
        performance_stats['hdf5_save'] = save_time
        print()

        # 第六步：VTK转换
        print("Step 6: VTK格式转换")
        output_vts = os.path.join(base_dir, "matrix_data", "perf_analysis_48x48x48.vts")

        def convert_vtk():
            storage.convert_to_vtk(output_h5, output_vts)

        _, vtk_time = time_function(
            convert_vtk, "VTK格式转换"
        )
        performance_stats['vtk_conversion'] = vtk_time
        print()

        # 计算总时间
        total_time = time.time() - total_start

        # 性能分析报告
        print("="*60)
        print("性能分析报告")
        print("="*60)

        # 按耗时排序
        sorted_stats = sorted(performance_stats.items(), key=lambda x: x[1], reverse=True)

        print(f"总执行时间: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
        print(f"网格点数: {np.prod(grid_size):,}")
        print(f"平均每点处理时间: {total_time/np.prod(grid_size)*1000:.3f}毫秒")
        print()

        print("各步骤耗时排名:")
        total_accounted = sum(performance_stats.values())
        for i, (step, time_taken) in enumerate(sorted_stats, 1):
            percentage = (time_taken / total_time) * 100
            time_per_point = time_taken / np.prod(grid_size) * 1000  # 毫秒

            print(f"{i}. {step:25s}: {time_taken:6.2f}秒 ({time_taken/60:4.1f}分钟) "
                  f"({percentage:5.1f}%) - {time_per_point:6.3f}毫秒/点")

        print()
        print(f"已统计步骤总计: {total_accounted:.2f}秒 ({total_accounted/total_time*100:.1f}%)")
        print(f"其他开销: {total_time-total_accounted:.2f}秒 ({(total_time-total_accounted)/total_time*100:.1f}%)")

        # 性能瓶颈识别
        print()
        print("性能瓶颈分析:")
        if sorted_stats[0][1] > total_time * 0.5:
            bottleneck = sorted_stats[0]
            print(f"🔴 主要瓶颈: {bottleneck[0]} (占用{bottleneck[1]/total_time*100:.1f}%的时间)")

        if sorted_stats[0][0] == 'sdf_computation':
            print("   - SDF计算是最大性能瓶颈")
            print("   - 建议: 使用并行计算、GPU加速或优化算法")
        elif sorted_stats[0][0] == 'interpolation':
            print("   - 插值计算是最大性能瓶颈")
            print("   - 建议: 使用LinearNDInterpolator预构建优化")

        # 与15分钟的对比
        expected_time = 15 * 60  # 15分钟
        print()
        print("性能对比:")
        print(f"用户报告时间: {expected_time/60:.1f}分钟")
        print(f"当前测试时间: {total_time/60:.1f}分钟")
        if total_time < expected_time:
            speedup = expected_time / total_time
            print(f"性能提升: {speedup:.1f}x 倍")
        else:
            slowdown = total_time / expected_time
            print(f"性能下降: {slowdown:.1f}x 倍")

        return True

    except Exception as e:
        print(f"\n[ERROR] 性能分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_performance_analysis()
    if not success:
        sys.exit(1)