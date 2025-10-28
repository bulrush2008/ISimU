"""
ISimU 快速开始示例

演示基本使用方法
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def quick_start():
    """快速开始示例"""
    print("=== ISimU Quick Start ===\n")

    # 检查数据文件
    vtm_file = "../Data/vessel.000170.vtm"
    if not os.path.exists(vtm_file):
        print(f"[ERROR] Data file not found: {vtm_file}")
        print("Please ensure data files are in the correct location")
        return

    print("[OK] Data file found")

    # 导入模块
    try:
        from data_reader import VTKReader
        from interpolator import GridInterpolator
        from hdf5_storage import HDF5Storage
        print("[OK] Modules imported successfully")
    except ImportError as e:
        print(f"[ERROR] Module import failed: {e}")
        print("Please install dependencies: uv sync")
        return

    try:
        # 1. 读取VTK文件
        print("\n1. Reading VTK file...")
        reader = VTKReader()
        data = reader.read_vtm(vtm_file)
        fields = reader.get_available_fields(data)
        print(f"   [OK] Successfully read, found fields: {fields}")

        # 2. 插值到笛卡尔网格
        print("\n2. Interpolating to Cartesian grid...")
        interpolator = GridInterpolator(grid_size=(16, 16, 16))  # 小网格用于快速测试
        result = interpolator.interpolate(data, fields[:1])  # 只插值第一个场变量
        print(f"   [OK] Interpolation completed, grid size: {result['grid_size']}")

        # 3. 保存为HDF5
        print("\n3. Saving to HDF5 format...")
        storage = HDF5Storage()
        output_path = "../Data/quick_start_output.h5"
        storage.save(result, output_path)
        print(f"   [OK] Save completed: {output_path}")

        print("\n[SUCCESS] Quick start example completed!")
        print(f"Output file: {output_path}")
        print("You can use ParaView to open the generated VTK file for visualization")

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_start()