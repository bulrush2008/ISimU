"""
VMR数据集完整插值测试
使用VTP几何数据进行32x32x32网格插值，输出HDF5和VTS文件
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
from interpolator_optimized import OptimizedGridInterpolator
from hdf5_storage import HDF5Storage
from sdf_utils import create_sdf_from_vmr_case, create_sdf_from_vtk_data


def load_vmr_case(case_id: str, base_dir: str = None):
    """
    加载VMR病例的几何和流场数据

    Args:
        case_id: VMR病例ID
        base_dir: 项目根目录

    Returns:
        (geometry_data, flow_data) 或 None
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载配置文件
    config_file = os.path.join(base_dir, "data_VMR", "geo-flow.json")

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        if case_id not in config:
            print(f"[ERROR] Case ID {case_id} not found in VMR config")
            return None

        case_data = config[case_id]
        geometry_path = os.path.join(base_dir, case_data['geo'])
        flow_path = os.path.join(base_dir, case_data['flow'])

        print(f"Loading VMR case {case_id}:")
        print(f"  - Geometry: {geometry_path}")
        print(f"  - Flow: {flow_path}")

        # 验证文件存在
        if not os.path.exists(geometry_path):
            print(f"  [ERROR] Geometry file not found: {geometry_path}")
            return None

        if not os.path.exists(flow_path):
            print(f"  [ERROR] Flow file not found: {flow_path}")
            return None

        # 读取流场数据
        reader = VTKReader()
        flow_data = reader.read_vtu(flow_path)

        print(f"  [OK] Flow data loaded:")
        print(f"    - Points: {flow_data['num_points']:,}")
        print(f"    - Cells: {flow_data['num_cells']:,}")

        if 'point_data' in flow_data:
            fields = list(flow_data['point_data'].keys())
            print(f"    - Fields: {fields}")

        return geometry_path, flow_data

    except Exception as e:
        print(f"[ERROR] Failed to load VMR case: {e}")
        return None


def main():
    """主测试函数"""
    print("=== VMR Data Interpolation Test ===")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 配置参数
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
    case_id = "0007_H_AO_H"  # 使用第一个VMR病例
    grid_size = (32, 32, 32)

    # 输出文件路径
    output_dir = os.path.join(base_dir, "data_matrix")
    os.makedirs(output_dir, exist_ok=True)

    output_h5 = os.path.join(output_dir, f"vmr_{case_id}_32x32x32.h5")
    output_vts = os.path.join(output_dir, f"vmr_{case_id}_32x32x32.vts")

    print(f"Configuration:")
    print(f"  - VMR Case: {case_id}")
    print(f"  - Grid size: {grid_size} ({np.prod(grid_size):,} points)")
    print(f"  - Output HDF5: {output_h5}")
    print(f"  - Output VTS: {output_vts}")
    print()

    try:
        # Step 1: 加载VMR数据
        print("Step 1: Loading VMR data...")
        geometry_path, vtu_data = load_vmr_case(case_id, base_dir)

        if geometry_path is None or vtu_data is None:
            print("  [ERROR] Failed to load VMR data")
            return False

        print("  [OK] VMR data loaded successfully")
        print()

        # Step 2: 创建SDF计算器
        print("Step 2: Creating SDF calculator from VTP geometry...")
        start_time = datetime.now()

        # 创建基于VTP几何的SDF计算器
        sdf = create_sdf_from_vmr_case(case_id)

        if sdf is None:
            print("  [ERROR] Failed to create SDF calculator")
            return False

        print(f"  [OK] SDF calculator created successfully")
        print(f"    - Vertices: {len(sdf.vertices):,}")
        print(f"    - Faces: {len(sdf.faces):,}")
        print(f"    - Geometry source: {getattr(sdf, 'geometry_source', 'Unknown')}")
        print()

        # Step 3: 创建插值器并设置SDF
        print("Step 3: Setting up interpolator...")
        interpolator = OptimizedGridInterpolator(
            grid_size=grid_size,
            method='linear',
            out_of_domain_value=0.0,
            use_sdf=True,
            batch_size=10000
        )

        # 手动设置SDF计算器以使用VTP数据
        interpolator.sdf_calculator = sdf

        print(f"  [OK] Interpolator configured:")
        print(f"    - Grid size: {interpolator.grid_size}")
        print(f"    - Method: {interpolator.method}")
        print(f"    - SDF enabled: {interpolator.use_sdf}")
        print()

        # Step 4: 执行插值
        print("Step 4: Performing interpolation...")
        if 'point_data' in vtu_data:
            fields = list(vtu_data['point_data'].keys())
        else:
            fields = []

        if not fields:
            print("  [ERROR] No fields found for interpolation")
            return False

        print(f"  Fields to interpolate: {fields}")

        interpolated_data = interpolator.interpolate(vtu_data, fields=fields)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"  [OK] Interpolation completed in {duration:.1f} seconds")
        print(f"    - Grid size: {interpolated_data['grid_size']}")
        print(f"    - SDF used: {interpolated_data['sdf_used']}")
        print()

        # 显示插值结果统计
        print("Step 5: Interpolation results summary:")
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
            print(f"    Shape: {field_info['shape']}")
            print(f"    Range: [{field_info['min']:.3e}, {field_info['max']:.3e}]")
            print(f"    Mean: {field_info['mean']:.3e}")
            print(f"    NaN count: {field_info['nan_count']}")
            print(f"    Zero count: {field_info['zero_count']}")
        print()

        # Step 6: 保存HDF5文件
        print("Step 6: Saving HDF5 file...")
        storage = HDF5Storage()

        metadata = {
            'source_case': case_id,
            'geometry_source': getattr(sdf, 'geometry_source', 'Unknown'),
            'geometry_file': geometry_path,
            'flow_file': 'data_VMR/' + case_id + '/Simulations/*/mesh-complete/initial.vtu',
            'grid_size': interpolated_data['grid_size'],
            'interpolation_method': 'linear',
            'use_sdf': interpolated_data['sdf_used'],
            'processing_time_seconds': duration,
            'total_points': int(np.prod(grid_size)),
            'creation_time': datetime.now().isoformat(),
            'description': f'{case_id} VMR case interpolation with VTP geometry, {grid_size[0]}x{grid_size[1]}x{grid_size[2]} grid'
        }

        storage.save(interpolated_data, output_h5, metadata=metadata)
        print(f"  [OK] HDF5 file saved: {output_h5}")

        # 显示文件大小
        file_size = os.path.getsize(output_h5) / (1024 * 1024)  # MB
        print(f"    File size: {file_size:.2f} MB")
        print()

        # Step 7: 转换为VTS格式
        print("Step 7: Converting to VTS format...")
        try:
            storage.convert_to_vtk(output_h5, output_vts)
            print(f"  [OK] VTS file saved: {output_vts}")

            # 显示VTS文件大小
            vts_size = os.path.getsize(output_vts) / (1024 * 1024)  # MB
            print(f"    File size: {vts_size:.2f} MB")
        except Exception as e:
            print(f"  [WARNING] VTS conversion failed: {e}")
        print()

        # Step 8: 性能统计
        print("Step 8: Performance statistics:")
        points_per_second = np.prod(grid_size) / duration
        print(f"  - Processing speed: {points_per_second:.0f} points/second")
        print(f"  - Memory efficiency: {file_size/np.prod(grid_size)*1024:.2f} KB/1k points")
        print()

        # 完成信息
        print("=== VMR Interpolation Completed Successfully ===")
        print(f"VMR Case: {case_id}")
        print(f"Grid size: {grid_size[0]}x{grid_size[1]}x{grid_size[2]}")
        print(f"Processing time: {duration:.1f} seconds")
        print(f"Output files:")
        print(f"  - HDF5: {output_h5}")
        print(f"  - VTS:  {output_vts}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return True

    except KeyboardInterrupt:
        print("\n[INFO] User interrupted operation")
        return False
    except Exception as e:
        print(f"\n[ERROR] VMR interpolation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)