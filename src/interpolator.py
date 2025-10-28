"""
网格插值模块

将非结构网格数据插值到均匀笛卡尔网格
"""

import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
from typing import Dict, Any, Tuple, Optional, List
import warnings


class GridInterpolator:
    """网格插值器"""

    def __init__(self,
                 grid_size: Tuple[int, int, int] = (64, 64, 64),
                 bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
                 method: str = 'linear'):
        """
        初始化插值器

        Args:
            grid_size: 笛卡尔网格尺寸 (nx, ny, nz)
            bounds: 边界范围 (xmin, xmax, ymin, ymax, zmin, zmax)
            method: 插值方法 ('linear', 'nearest', 'cubic')
        """
        self.grid_size = grid_size
        self.method = method
        self.bounds = bounds
        self.cartesian_grid = None
        self.interpolators = {}

    def setup_cartesian_grid(self, vertices: np.ndarray) -> None:
        """
        根据顶点坐标设置笛卡尔网格

        Args:
            vertices: 非结构网格顶点坐标
        """
        if self.bounds is None:
            # 自动计算边界
            self.bounds = (
                np.min(vertices[:, 0]), np.max(vertices[:, 0]),  # x bounds
                np.min(vertices[:, 1]), np.max(vertices[:, 1]),  # y bounds
                np.min(vertices[:, 2]), np.max(vertices[:, 2])   # z bounds
            )

        # 创建均匀笛卡尔网格
        x = np.linspace(self.bounds[0], self.bounds[1], self.grid_size[0])
        y = np.linspace(self.bounds[2], self.bounds[3], self.grid_size[1])
        z = np.linspace(self.bounds[4], self.bounds[5], self.grid_size[2])

        # 创建3D网格坐标
        self.cartesian_grid = np.meshgrid(x, y, z, indexing='ij')

        print(f"Cartesian grid setup completed:")
        print(f"  - Grid size: {self.grid_size}")
        print(f"  - X range: [{self.bounds[0]:.3f}, {self.bounds[1]:.3f}]")
        print(f"  - Y range: [{self.bounds[2]:.3f}, {self.bounds[3]:.3f}]")
        print(f"  - Z range: [{self.bounds[4]:.3f}, {self.bounds[5]:.3f}]")

    def interpolate(self, vtk_data: Dict[str, Any],
                   fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行插值操作

        Args:
            vtk_data: VTK读取的数据
            fields: 需要插值的物理场变量列表

        Returns:
            插值后的笛卡尔网格数据
        """
        # 提取第一个非空数据块
        source_block = None
        for block in vtk_data['blocks']:
            if block['num_points'] > 0:
                source_block = block
                break

        if source_block is None:
            raise ValueError("没有找到有效的数据块")

        vertices = source_block['vertices']
        point_data = source_block['point_data']

        # 设置笛卡尔网格
        self.setup_cartesian_grid(vertices)

        # 准备插值点坐标
        query_points = np.column_stack([
            self.cartesian_grid[0].ravel(),
            self.cartesian_grid[1].ravel(),
            self.cartesian_grid[2].ravel()
        ])

        # 确定要插值的场
        if fields is None:
            fields = list(point_data.keys())

        print(f"Starting interpolation, fields: {fields}")

        result = {
            'grid_size': self.grid_size,
            'bounds': self.bounds,
            'grid_coordinates': {
                'x': self.cartesian_grid[0],
                'y': self.cartesian_grid[1],
                'z': self.cartesian_grid[2]
            },
            'fields': {}
        }

        # 对每个场变量进行插值
        for field_name in fields:
            if field_name not in point_data:
                warnings.warn(f"场变量 '{field_name}' 不存在，跳过")
                continue

            field_values = point_data[field_name]
            interpolated_data = self._interpolate_field(
                vertices, field_values, query_points, field_name
            )

            # 重新整形为网格形状
            if len(interpolated_data.shape) == 1:  # 标量场
                interpolated_data = interpolated_data.reshape(self.grid_size)
            else:  # 矢量场或张量场
                new_shape = self.grid_size + (interpolated_data.shape[-1],)
                interpolated_data = interpolated_data.reshape(new_shape)

            result['fields'][field_name] = interpolated_data
            print(f"  [OK] {field_name}: {field_values.shape} -> {interpolated_data.shape}")

        return result

    def _interpolate_field(self,
                          vertices: np.ndarray,
                          values: np.ndarray,
                          query_points: np.ndarray,
                          field_name: str) -> np.ndarray:
        """
        插值单个场变量

        Args:
            vertices: 源网格顶点坐标
            values: 源网格上的场变量值
            query_points: 查询点坐标
            field_name: 场变量名称

        Returns:
            插值后的场变量值
        """
        try:
            # 使用scipy的griddata进行插值
            if self.method == 'linear':
                interpolated = griddata(
                    vertices, values, query_points,
                    method='linear', fill_value=0.0
                )
            elif self.method == 'nearest':
                interpolated = griddata(
                    vertices, values, query_points,
                    method='nearest', fill_value=0.0
                )
            elif self.method == 'cubic':
                interpolated = griddata(
                    vertices, values, query_points,
                    method='cubic', fill_value=0.0
                )
            else:
                raise ValueError(f"不支持的插值方法: {self.method}")

        except Exception as e:
            print(f"插值失败 {field_name}: {e}")
            # 回退到最近邻插值
            interpolated = griddata(
                vertices, values, query_points,
                method='nearest', fill_value=0.0
            )

        return interpolated

    def get_interpolation_statistics(self, interpolated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取插值统计信息

        Args:
            interpolated_data: 插值后的数据

        Returns:
            统计信息
        """
        stats = {
            'total_points': np.prod(self.grid_size),
            'grid_size': self.grid_size,
            'bounds': self.bounds,
            'field_statistics': {}
        }

        for field_name, field_data in interpolated_data['fields'].items():
            field_stats = {
                'shape': field_data.shape,
                'dtype': field_data.dtype,
                'min': float(np.min(field_data)),
                'max': float(np.max(field_data)),
                'mean': float(np.mean(field_data)),
                'std': float(np.std(field_data)),
                'nan_count': int(np.sum(np.isnan(field_data))),
                'zero_count': int(np.sum(field_data == 0))
            }
            stats['field_statistics'][field_name] = field_stats

        return stats


def test_interpolator():
    """测试插值功能"""
    from data_reader import VTKReader

    print("=== 插值功能测试 ===")

    # 读取VTK数据
    reader = VTKReader()
    vtm_file = "../Data/vessel.000170.vtm"

    try:
        vtk_data = reader.read_vtm(vtm_file)
        print(f"✓ 成功读取VTK数据")
    except Exception as e:
        print(f"✗ 读取VTK失败: {e}")
        return

    # 创建插值器
    interpolator = GridInterpolator(
        grid_size=(32, 32, 32),  # 较小的测试网格
        method='linear'
    )

    try:
        # 执行插值
        result = interpolator.interpolate(vtk_data)
        print(f"✓ 插值完成")

        # 显示统计信息
        stats = interpolator.get_interpolation_statistics(result)
        print(f"插值统计:")
        print(f"  - 总网格点数: {stats['total_points']:,}")
        print(f"  - 网格尺寸: {stats['grid_size']}")

        for field_name, field_stats in stats['field_statistics'].items():
            print(f"  - {field_name}:")
            print(f"    形状: {field_stats['shape']}")
            print(f"    范围: [{field_stats['min']:.3e}, {field_stats['max']:.3e}]")
            print(f"    均值: {field_stats['mean']:.3e}")
            print(f"    NaN点数: {field_stats['nan_count']}")
            print(f"    零值点数: {field_stats['zero_count']}")

    except Exception as e:
        print(f"✗ 插值失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_interpolator()