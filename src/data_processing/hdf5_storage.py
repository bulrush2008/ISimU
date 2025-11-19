"""
HDF5数据存储模块

将插值后的矩阵数据存储为HDF5格式
"""

import h5py
import numpy as np
from typing import Dict, Any, Optional, List
import json
from datetime import datetime


class HDF5Storage:
    """HDF5数据存储器"""

    def __init__(self):
        self.file_handle = None

    def save(self, data: Dict[str, Any],
             output_path: str,
             metadata: Optional[Dict[str, Any]] = None,
             compression: str = 'gzip') -> None:
        """
        保存插值数据到HDF5文件

        Args:
            data: 插值后的数据
            output_path: 输出文件路径
            metadata: 元数据信息
            compression: 压缩算法 ('gzip', 'lzf', 'szip')
        """
        print(f"Saving data to: {output_path}")

        with h5py.File(output_path, 'w') as f:
            # 保存基本信息
            f.attrs['creation_time'] = datetime.now().isoformat()
            f.attrs['version'] = '1.0'

            # 保存网格信息
            grid_group = f.create_group('grid')
            grid_group.attrs['grid_size'] = data['grid_size']
            grid_group.attrs['bounds'] = data['bounds']

            # 保存网格坐标
            grid_coords = data['grid_coordinates']
            grid_group.create_dataset('x', data=grid_coords['x'], compression=compression)
            grid_group.create_dataset('y', data=grid_coords['y'], compression=compression)
            grid_group.create_dataset('z', data=grid_coords['z'], compression=compression)

            # 保存物理场数据
            fields_group = f.create_group('fields')
            for field_name, field_data in data['fields'].items():
                dataset = fields_group.create_dataset(
                    field_name,
                    data=field_data,
                    compression=compression,
                    compression_opts=9 if compression == 'gzip' else None
                )
                dataset.attrs['shape'] = field_data.shape
                dataset.attrs['dtype'] = str(field_data.dtype)

            # 保存元数据
            if metadata:
                metadata_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        metadata_group.attrs[key] = value
                    elif isinstance(value, (list, tuple)):
                        metadata_group.attrs[key] = str(value)
                    elif isinstance(value, dict):
                        metadata_group.create_dataset(key, data=json.dumps(value))

        print(f"[OK] Data save completed")

    def load(self, file_path: str) -> Dict[str, Any]:
        """
        从HDF5文件加载数据

        Args:
            file_path: HDF5文件路径

        Returns:
            加载的数据
        """
        print(f"Loading data from: {file_path}")

        data = {}

        with h5py.File(file_path, 'r') as f:
            # 读取基本信息
            data['creation_time'] = f.attrs.get('creation_time', '')
            data['version'] = f.attrs.get('version', '1.0')

            # 读取网格信息
            grid_group = f['grid']
            data['grid_size'] = tuple(grid_group.attrs['grid_size'])
            data['bounds'] = tuple(grid_group.attrs['bounds'])

            # 读取网格坐标
            data['grid_coordinates'] = {
                'x': grid_group['x'][:],
                'y': grid_group['y'][:],
                'z': grid_group['z'][:]
            }

            # 读取物理场数据
            data['fields'] = {}
            fields_group = f['fields']
            for field_name in fields_group.keys():
                data['fields'][field_name] = fields_group[field_name][:]

            # 读取元数据
            if 'metadata' in f:
                data['metadata'] = {}
                metadata_group = f['metadata']
                for key in metadata_group.attrs.keys():
                    data['metadata'][key] = metadata_group.attrs[key]

                for dataset_name in metadata_group.keys():
                    if hasattr(metadata_group[dataset_name], 'attrs'):
                        data['metadata'][dataset_name] = json.loads(
                            metadata_group[dataset_name][()].decode('utf-8')
                        )

        print(f"[OK] Data load completed")
        return data

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        获取HDF5文件信息

        Args:
            file_path: HDF5文件路径

        Returns:
            文件信息
        """
        info = {}

        with h5py.File(file_path, 'r') as f:
            info['creation_time'] = f.attrs.get('creation_time', '')
            info['version'] = f.attrs.get('version', '1.0')

            # 网格信息
            if 'grid' in f:
                grid_group = f['grid']
                info['grid_size'] = tuple(grid_group.attrs['grid_size'])
                info['bounds'] = tuple(grid_group.attrs['bounds'])

            # 场变量信息
            if 'fields' in f:
                fields_group = f['fields']
                info['fields'] = {}
                total_size = 0

                for field_name in fields_group.keys():
                    dataset = fields_group[field_name]
                    field_info = {
                        'shape': dataset.shape,
                        'dtype': str(dataset.dtype),
                        'size_mb': dataset.nbytes / (1024 * 1024)
                    }
                    info['fields'][field_name] = field_info
                    total_size += field_info['size_mb']

                info['total_data_size_mb'] = total_size

        return info

    def convert_to_vtk(self, hdf5_path: str, vtk_path: str) -> None:
        """
        将HDF5数据转换为VTK格式用于可视化

        Args:
            hdf5_path: HDF5文件路径
            vtk_path: 输出VTK文件路径
        """
        try:
            import vtk
            from vtk.util import numpy_support
        except ImportError:
            print("[ERROR] VTK library not installed, cannot convert to VTK format")
            return

        print(f"Converting HDF5 to VTK: {hdf5_path} -> {vtk_path}")

        # 加载HDF5数据
        data = self.load(hdf5_path)

        # 创建VTK结构化网格
        grid_size = data['grid_size']
        vtk_grid = vtk.vtkStructuredGrid()
        vtk_grid.SetDimensions(grid_size)

        # 创建网格点
        x = data['grid_coordinates']['x'].ravel()
        y = data['grid_coordinates']['y'].ravel()
        z = data['grid_coordinates']['z'].ravel()

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(len(x))

        for i in range(len(x)):
            points.SetPoint(i, x[i], y[i], z[i])

        vtk_grid.SetPoints(points)

        # 添加物理场数据
        for field_name, field_data in data['fields'].items():
            if len(field_data.shape) == 3:  # 标量场
                flat_data = field_data.ravel()
                vtk_array = numpy_support.numpy_to_vtk(
                    flat_data, deep=True, array_type=vtk.VTK_FLOAT
                )
                vtk_array.SetName(field_name)
                vtk_grid.GetPointData().AddArray(vtk_array)

            elif len(field_data.shape) == 4:  # 矢量场
                flat_data = field_data.reshape(-1, field_data.shape[-1])
                vtk_array = numpy_support.numpy_to_vtk(
                    flat_data, deep=True, array_type=vtk.VTK_FLOAT
                )
                vtk_array.SetName(field_name)
                vtk_array.SetNumberOfComponents(field_data.shape[-1])
                vtk_grid.GetPointData().AddArray(vtk_array)

        # 写入VTK文件
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(vtk_path)
        writer.SetInputData(vtk_grid)
        writer.Write()

        print(f"[OK] VTK file saved: {vtk_path}")


def test_hdf5_storage():
    """测试HDF5存储功能"""
    print("=== HDF5存储功能测试 ===")

    # 创建测试数据
    test_data = {
        'grid_size': (10, 10, 10),
        'bounds': (0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        'grid_coordinates': {
            'x': np.random.rand(10, 10, 10),
            'y': np.random.rand(10, 10, 10),
            'z': np.random.rand(10, 10, 10)
        },
        'fields': {
            'velocity': np.random.randn(10, 10, 10, 3),
            'pressure': np.random.rand(10, 10, 10),
            'temperature': np.random.rand(10, 10, 10) * 100
        }
    }

    # 测试数据
    metadata = {
        'source_file': 'test_vtm.vtm',
        'interpolation_method': 'linear',
        'description': '测试数据'
    }

    storage = HDF5Storage()

    try:
        # 保存数据
        output_path = "test_data.h5"
        storage.save(test_data, output_path, metadata)
        print(f"✓ 保存测试数据成功")

        # 加载数据
        loaded_data = storage.load(output_path)
        print(f"✓ 加载数据成功")

        # 验证数据一致性
        assert loaded_data['grid_size'] == test_data['grid_size']
        assert loaded_data['bounds'] == test_data['bounds']
        print(f"✓ 数据一致性验证通过")

        # 显示文件信息
        file_info = storage.get_file_info(output_path)
        print(f"文件信息:")
        print(f"  - 网格尺寸: {file_info['grid_size']}")
        print(f"  - 数据大小: {file_info['total_data_size_mb']:.2f} MB")
        for field_name, field_info in file_info['fields'].items():
            print(f"  - {field_name}: {field_info['shape']} ({field_info['size_mb']:.2f} MB)")

        # 测试VTK转换（如果VTK可用）
        try:
            vtk_output = "test_data.vts"
            storage.convert_to_vtk(output_path, vtk_output)
            print(f"✓ VTK转换成功")
        except Exception as e:
            print(f"  VTK转换跳过: {e}")

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_hdf5_storage()