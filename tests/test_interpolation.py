"""
插值功能测试

测试VTK读取、网格插值和HDF5存储的完整流程
"""

import unittest
import numpy as np
import os
import sys

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_reader import VTKReader
from interpolator import GridInterpolator
from hdf5_storage import HDF5Storage


class TestInterpolation(unittest.TestCase):
    """插值功能测试类"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'Data')
        cls.vtm_file = os.path.join(cls.test_data_dir, 'vessel.000170.vtm')
        cls.output_h5 = os.path.join(cls.test_data_dir, 'test_output.h5')

    def setUp(self):
        """每个测试方法的初始化"""
        self.reader = VTKReader()
        self.interpolator = GridInterpolator(grid_size=(16, 16, 16))
        self.storage = HDF5Storage()

    def test_vtk_reader(self):
        """测试VTK读取功能"""
        print("\n测试VTK读取功能...")

        if not os.path.exists(self.vtm_file):
            self.skipTest("测试VTM文件不存在")

        # 读取VTM文件
        data = self.reader.read_vtm(self.vtm_file)

        # 验证数据结构
        self.assertIn('blocks', data)
        self.assertIn('num_blocks', data)
        self.assertGreater(data['num_blocks'], 0)

        # 验证至少有一个有效数据块
        valid_blocks = [block for block in data['blocks'] if block['num_points'] > 0]
        self.assertGreater(len(valid_blocks), 0)

        # 验证数据块结构
        first_block = valid_blocks[0]
        self.assertIn('vertices', first_block)
        self.assertIn('num_points', first_block)
        self.assertIn('point_data', first_block)

        # 验证顶点数据
        vertices = first_block['vertices']
        self.assertEqual(len(vertices.shape), 2)
        self.assertEqual(vertices.shape[1], 3)  # x, y, z 坐标

        print(f"  ✓ 成功读取 {data['num_blocks']} 个数据块")
        print(f"  ✓ 第一个数据块包含 {first_block['num_points']} 个点")

    def test_interpolator(self):
        """测试插值功能"""
        print("\n测试插值功能...")

        if not os.path.exists(self.vtm_file):
            self.skipTest("测试VTM文件不存在")

        # 读取数据
        vtk_data = self.reader.read_vtm(self.vtm_file)

        # 获取可用场变量
        available_fields = self.reader.get_available_fields(vtk_data)
        if not available_fields:
            self.skipTest("没有可用的场变量")

        # 选择前2个场变量进行测试
        test_fields = available_fields[:2]

        # 执行插值
        interpolated_data = self.interpolator.interpolate(vtk_data, test_fields)

        # 验证插值结果结构
        self.assertIn('grid_size', interpolated_data)
        self.assertIn('bounds', interpolated_data)
        self.assertIn('grid_coordinates', interpolated_data)
        self.assertIn('fields', interpolated_data)

        # 验证网格尺寸
        self.assertEqual(interpolated_data['grid_size'], (16, 16, 16))

        # 验证场变量
        self.assertEqual(len(interpolated_data['fields']), len(test_fields))

        for field_name in test_fields:
            self.assertIn(field_name, interpolated_data['fields'])
            field_data = interpolated_data['fields'][field_name]

            # 验证数据形状
            expected_shape = (16, 16, 16)
            if len(field_data.shape) == 4:  # 矢量场
                expected_shape = expected_shape + (field_data.shape[-1],)

            self.assertEqual(field_data.shape, expected_shape)

            # 验证数据类型
            self.assertTrue(np.issubdtype(field_data.dtype, np.floating))

            # 验证没有全为NaN
            self.assertLess(np.sum(np.isnan(field_data)), field_data.size * 0.1)

        print(f"  ✓ 成功插值 {len(test_fields)} 个场变量")
        print(f"  ✓ 网格尺寸: {interpolated_data['grid_size']}")

    def test_hdf5_storage(self):
        """测试HDF5存储功能"""
        print("\n测试HDF5存储功能...")

        # 创建测试数据
        test_data = {
            'grid_size': (8, 8, 8),
            'bounds': (0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            'grid_coordinates': {
                'x': np.linspace(0, 1, 8).reshape(8, 1, 1).repeat(8, axis=1).repeat(8, axis=2),
                'y': np.linspace(0, 1, 8).reshape(1, 8, 1).repeat(8, axis=0).repeat(8, axis=2),
                'z': np.linspace(0, 1, 8).reshape(1, 1, 8).repeat(8, axis=0).repeat(8, axis=1),
            },
            'fields': {
                'test_scalar': np.random.rand(8, 8, 8),
                'test_vector': np.random.randn(8, 8, 8, 3),
            }
        }

        metadata = {
            'test': True,
            'description': '测试数据'
        }

        # 测试保存
        self.storage.save(test_data, self.output_h5, metadata)
        self.assertTrue(os.path.exists(self.output_h5))

        # 测试加载
        loaded_data = self.storage.load(self.output_h5)

        # 验证数据一致性
        self.assertEqual(loaded_data['grid_size'], test_data['grid_size'])
        self.assertEqual(loaded_data['bounds'], test_data['bounds'])

        # 验证场变量
        for field_name in test_data['fields']:
            self.assertIn(field_name, loaded_data['fields'])
            np.testing.assert_array_almost_equal(
                loaded_data['fields'][field_name],
                test_data['fields'][field_name]
            )

        print(f"  ✓ 成功保存和加载测试数据")

        # 清理测试文件
        if os.path.exists(self.output_h5):
            os.remove(self.output_h5)

    def test_complete_pipeline(self):
        """测试完整处理流程"""
        print("\n测试完整处理流程...")

        if not os.path.exists(self.vtm_file):
            self.skipTest("测试VTM文件不存在")

        try:
            # 读取VTK
            vtk_data = self.reader.read_vtm(self.vtm_file)

            # 插值
            available_fields = self.reader.get_available_fields(vtk_data)
            if not available_fields:
                self.skipTest("没有可用的场变量")

            test_fields = available_fields[:1]  # 只测试一个场变量以节省时间
            interpolated_data = self.interpolator.interpolate(vtk_data, test_fields)

            # 保存HDF5
            metadata = {
                'test_complete_pipeline': True,
                'source_file': self.vtm_file,
                'interpolated_fields': test_fields
            }
            self.storage.save(interpolated_data, self.output_h5, metadata)

            # 验证输出文件
            self.assertTrue(os.path.exists(self.output_h5))

            # 验证文件信息
            file_info = self.storage.get_file_info(self.output_h5)
            self.assertIn('grid_size', file_info)
            self.assertIn('fields', file_info)

            print(f"  ✓ 完整流程测试成功")
            print(f"  ✓ 输出文件大小: {file_info.get('total_data_size_mb', 0):.2f} MB")

            # 清理测试文件
            if os.path.exists(self.output_h5):
                os.remove(self.output_h5)

        except Exception as e:
            self.fail(f"完整流程测试失败: {e}")

    def test_interpolation_quality(self):
        """测试插值质量"""
        print("\n测试插值质量...")

        if not os.path.exists(self.vtm_file):
            self.skipTest("测试VTM文件不存在")

        # 读取数据
        vtk_data = self.reader.read_vtm(self.vtm_file)
        available_fields = self.reader.get_available_fields(vtk_data)

        if not available_fields:
            self.skipTest("没有可用的场变量")

        # 使用小网格进行测试
        small_interpolator = GridInterpolator(grid_size=(8, 8, 8))
        interpolated_data = small_interpolator.interpolate(vtk_data, available_fields[:1])

        # 获取统计信息
        stats = small_interpolator.get_interpolation_statistics(interpolated_data)

        # 验证统计信息
        self.assertIn('field_statistics', stats)

        for field_name, field_stats in stats['field_statistics'].items():
            # 验证基本统计量
            self.assertIn('min', field_stats)
            self.assertIn('max', field_stats)
            self.assertIn('mean', field_stats)
            self.assertIn('std', field_stats)

            # 验证数据质量
            nan_ratio = field_stats['nan_count'] / field_stats['shape'][0]
            self.assertLess(nan_ratio, 0.5, f"场变量 {field_name} NaN比例过高: {nan_ratio}")

            print(f"  ✓ {field_name}: 范围[{field_stats['min']:.3e}, {field_stats['max']:.3e}], "
                  f"NaN比例: {nan_ratio:.2%}")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)