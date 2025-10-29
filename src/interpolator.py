"""
网格插值模块

将非结构网格数据插值到均匀笛卡尔网格
使用SDF判断血管内外的位置
"""

import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
from typing import Dict, Any, Tuple, Optional, List
import warnings
from sdf_utils import VascularSDF, create_sdf_from_vtk_data


class GridInterpolator:
    """网格插值器"""

    def __init__(self,
                 grid_size: Tuple[int, int, int] = (128, 128, 128),
                 bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
                 method: str = 'linear',
                 out_of_domain_value: float = 0.0,
                 use_sdf: bool = True):
        """
        初始化插值器

        Args:
            grid_size: 笛卡尔网格尺寸 (nx, ny, nz)，默认为128x128x128
            bounds: 边界范围 (xmin, xmax, ymin, ymax, zmin, zmax)，默认使用原始数据范围
            method: 插值方法 ('linear', 'nearest', 'cubic')
            out_of_domain_value: 域外点的赋值，默认为0.0
            use_sdf: 是否使用SDF判断血管内外，默认为True
        """
        self.grid_size = grid_size
        self.method = method
        self.bounds = bounds
        self.out_of_domain_value = out_of_domain_value
        self.use_sdf = use_sdf
        self.cartesian_grid = None
        self.interpolators = {}
        self.sdf_calculator = None

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
        cell_data = source_block.get('cell_data', {})

        # 设置笛卡尔网格
        self.setup_cartesian_grid(vertices)

        # 准备插值点坐标
        query_points = np.column_stack([
            self.cartesian_grid[0].ravel(),
            self.cartesian_grid[1].ravel(),
            self.cartesian_grid[2].ravel()
        ])

        # 创建SDF计算器
        sdf_values = None
        if self.use_sdf:
            print("Creating SDF calculator...")
            self.sdf_calculator = create_sdf_from_vtk_data(vtk_data)
            if self.sdf_calculator is not None:
                print("  [OK] SDF calculator created successfully")
                # 计算SDF值
                print("  Computing SDF values for all grid points...")
                sdf_values = self.sdf_calculator.compute_sdf(query_points)

                # 判断血管内外的点
                inside_mask, outside_mask = self.sdf_calculator.get_inside_outside_mask(query_points)
                inside_count = np.sum(inside_mask)
                outside_count = np.sum(outside_mask)
                total_count = len(query_points)
                print(f"  - Inside vessel: {inside_count:,} ({inside_count/total_count*100:.1f}%)")
                print(f"  - Outside vessel: {outside_count:,} ({outside_count/total_count*100:.1f}%)")
                print(f"  - SDF range: [{np.min(sdf_values):.3e}, {np.max(sdf_values):.3e}]")
            else:
                print("  [WARNING] Failed to create SDF calculator, using all points")
                inside_mask = np.ones(len(query_points), dtype=bool)
                outside_mask = np.zeros(len(query_points), dtype=bool)
                # 创建虚拟SDF值（全部为正值，表示内部）
                sdf_values = np.ones(len(query_points)) * 1.0
        else:
            print("SDF disabled, using all points for interpolation")
            inside_mask = np.ones(len(query_points), dtype=bool)
            outside_mask = np.zeros(len(query_points), dtype=bool)
            # 创建虚拟SDF值
            sdf_values = np.ones(len(query_points)) * 1.0

        # 合并点数据和单元数据的字段名
        all_available_fields = list(point_data.keys()) + list(cell_data.keys())

        # 确定要插值的场
        if fields is None:
            fields = all_available_fields

        print(f"Starting interpolation, fields: {fields}")

        result = {
            'grid_size': self.grid_size,
            'bounds': self.bounds,
            'grid_coordinates': {
                'x': self.cartesian_grid[0],
                'y': self.cartesian_grid[1],
                'z': self.cartesian_grid[2]
            },
            'fields': {},
            'sdf_used': self.use_sdf and self.sdf_calculator is not None,
            'inside_point_count': np.sum(inside_mask),
            'outside_point_count': np.sum(outside_mask)
        }

        # 添加SDF值作为字段
        if sdf_values is not None:
            # 确保SDF值的符号正确：
            # - 正值：血管内部，距离血管壁的距离
            # - 负值：血管外部，距离血管壁的距离的负值
            sdf_field = sdf_values.reshape(self.grid_size)
            result['fields']['SDF'] = sdf_field
            print(f"  [OK] SDF: ({sdf_values.size}) -> {sdf_field.shape}")

            # 验证SDF值分布
            positive_count = np.sum(sdf_values > 0)
            negative_count = np.sum(sdf_values < 0)
            zero_count = np.sum(sdf_values == 0)
            print(f"    - Positive (inside): {positive_count} ({positive_count/len(sdf_values)*100:.1f}%)")
            print(f"    - Negative (outside): {negative_count} ({negative_count/len(sdf_values)*100:.1f}%)")
            print(f"    - Zero (on surface): {zero_count} ({zero_count/len(sdf_values)*100:.1f}%)")

            if positive_count > 0:
                print(f"    - Positive range: [{np.min(sdf_values[sdf_values > 0]):.6e}, {np.max(sdf_values[sdf_values > 0]):.6e}]")
            if negative_count > 0:
                print(f"    - Negative range: [{np.min(sdf_values[sdf_values < 0]):.6e}, {np.max(sdf_values[sdf_values < 0]):.6e}]")

        # 对每个场变量进行插值
        for field_name in fields:
            field_values = None
            data_source = None

            # 首先在点数据中查找
            if field_name in point_data:
                field_values = point_data[field_name]
                data_source = 'point_data'
            # 然后在单元数据中查找
            elif field_name in cell_data:
                field_values = cell_data[field_name]
                data_source = 'cell_data'
            else:
                warnings.warn(f"场变量 '{field_name}' 不存在，跳过")
                continue

            print(f"  Processing {field_name} from {data_source}")
            interpolated_data = self._interpolate_field_with_sdf(
                vertices, field_values, query_points, field_name, inside_mask, outside_mask
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
                    method='linear', fill_value=self.out_of_domain_value
                )
            elif self.method == 'nearest':
                interpolated = griddata(
                    vertices, values, query_points,
                    method='nearest', fill_value=self.out_of_domain_value
                )
            elif self.method == 'cubic':
                interpolated = griddata(
                    vertices, values, query_points,
                    method='cubic', fill_value=self.out_of_domain_value
                )
            else:
                raise ValueError(f"不支持的插值方法: {self.method}")

        except Exception as e:
            print(f"插值失败 {field_name}: {e}")
            # 回退到最近邻插值
            interpolated = griddata(
                vertices, values, query_points,
                method='nearest', fill_value=self.out_of_domain_value
            )

        return interpolated

    def _interpolate_field_with_sdf(self,
                                   vertices: np.ndarray,
                                   values: np.ndarray,
                                   query_points: np.ndarray,
                                   field_name: str,
                                   inside_mask: np.ndarray,
                                   outside_mask: np.ndarray) -> np.ndarray:
        """
        使用SDF判断进行插值（符合CLAUDE.md需求）

        Args:
            vertices: 源网格顶点坐标
            values: 源网格上的场变量值
            query_points: 查询点坐标
            field_name: 场变量名称
            inside_mask: 血管内部的点掩码
            outside_mask: 血管外部的点掩码

        Returns:
            插值后的场变量值
        """
        # 处理标量场和矢量场
        if len(values.shape) == 1:
            # 标量场
            return self._interpolate_scalar_field_with_sdf(
                vertices, values, query_points, field_name, inside_mask, outside_mask
            )
        else:
            # 矢量场
            return self._interpolate_vector_field_with_sdf(
                vertices, values, query_points, field_name, inside_mask, outside_mask
            )

    def _interpolate_scalar_field_with_sdf(self,
                                          vertices: np.ndarray,
                                          values: np.ndarray,
                                          query_points: np.ndarray,
                                          field_name: str,
                                          inside_mask: np.ndarray,
                                          outside_mask: np.ndarray) -> np.ndarray:
        """插值标量场"""
        # 初始化结果数组
        interpolated = np.full(len(query_points), self.out_of_domain_value, dtype=values.dtype)

        # 只对血管内部的点进行插值
        if np.sum(inside_mask) > 0:
            inside_points = query_points[inside_mask]

            try:
                # 使用scipy的griddata进行插值
                if self.method == 'linear':
                    inside_values = griddata(
                        vertices, values, inside_points,
                        method='linear', fill_value=self.out_of_domain_value
                    )
                elif self.method == 'nearest':
                    inside_values = griddata(
                        vertices, values, inside_points,
                        method='nearest', fill_value=self.out_of_domain_value
                    )
                elif self.method == 'cubic':
                    inside_values = griddata(
                        vertices, values, inside_points,
                        method='cubic', fill_value=self.out_of_domain_value
                    )
                else:
                    raise ValueError(f"不支持的插值方法: {self.method}")

                # 将插值结果放回原位
                interpolated[inside_mask] = inside_values

            except Exception as e:
                print(f"插值失败 {field_name}: {e}")
                # 回退到最近邻插值
                try:
                    inside_values = griddata(
                        vertices, values, inside_points,
                        method='nearest', fill_value=self.out_of_domain_value
                    )
                    interpolated[inside_mask] = inside_values
                except Exception as e2:
                    print(f"最近邻插值也失败 {field_name}: {e2}")
                    # 保持域外值不变

        # 血管外部的点保持域外值（-1）
        interpolated[outside_mask] = self.out_of_domain_value

        return interpolated

    def _interpolate_vector_field_with_sdf(self,
                                          vertices: np.ndarray,
                                          values: np.ndarray,
                                          query_points: np.ndarray,
                                          field_name: str,
                                          inside_mask: np.ndarray,
                                          outside_mask: np.ndarray) -> np.ndarray:
        """插值矢量场"""
        num_components = values.shape[1]
        num_points = len(query_points)

        # 初始化结果数组
        interpolated = np.full((num_points, num_components), self.out_of_domain_value, dtype=values.dtype)

        # 只对血管内部的点进行插值
        if np.sum(inside_mask) > 0:
            inside_points = query_points[inside_mask]

            # 对每个分量分别插值
            for component in range(num_components):
                component_values = values[:, component]

                try:
                    # 使用scipy的griddata进行插值
                    if self.method == 'linear':
                        inside_values = griddata(
                            vertices, component_values, inside_points,
                            method='linear', fill_value=self.out_of_domain_value
                        )
                    elif self.method == 'nearest':
                        inside_values = griddata(
                            vertices, component_values, inside_points,
                            method='nearest', fill_value=self.out_of_domain_value
                        )
                    elif self.method == 'cubic':
                        inside_values = griddata(
                            vertices, component_values, inside_points,
                            method='cubic', fill_value=self.out_of_domain_value
                        )
                    else:
                        raise ValueError(f"不支持的插值方法: {self.method}")

                    # 将插值结果放回原位
                    interpolated[inside_mask, component] = inside_values

                except Exception as e:
                    print(f"插值失败 {field_name}[{component}]: {e}")
                    # 回退到最近邻插值
                    try:
                        inside_values = griddata(
                            vertices, component_values, inside_points,
                            method='nearest', fill_value=self.out_of_domain_value
                        )
                        interpolated[inside_mask, component] = inside_values
                    except Exception as e2:
                        print(f"最近邻插值也失败 {field_name}[{component}]: {e2}")
                        # 保持域外值不变

        # 血管外部的点保持域外值（-1）
        interpolated[outside_mask] = self.out_of_domain_value

        return interpolated

    def interpolate_with_custom_methods(self, vtk_data: Dict[str, Any],
                                      fields: Optional[List[str]] = None,
                                      method_type: str = 'nearest') -> Dict[str, Any]:
        """
        使用自定义插值方法进行插值（符合CLAUDE.md需求）

        Args:
            vtk_data: VTK读取的数据
            fields: 需要插值的物理场变量列表
            method_type: 插值方法类型 ('nearest' 或 'average')
                        - 'nearest': 使用最近的网格直接赋值
                        - 'average': 使用临界的3个网格值的平均值

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

        print(f"Starting custom interpolation, method: {method_type}, fields: {fields}")

        result = {
            'grid_size': self.grid_size,
            'bounds': self.bounds,
            'grid_coordinates': {
                'x': self.cartesian_grid[0],
                'y': self.cartesian_grid[1],
                'z': self.cartesian_grid[2]
            },
            'fields': {},
            'interpolation_method': f'custom_{method_type}'
        }

        # 对每个场变量进行插值
        for field_name in fields:
            if field_name not in point_data:
                warnings.warn(f"场变量 '{field_name}' 不存在，跳过")
                continue

            field_values = point_data[field_name]
            interpolated_data = self._interpolate_field_custom(
                vertices, field_values, query_points, field_name, method_type
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

    def _interpolate_field_custom(self,
                                 vertices: np.ndarray,
                                 values: np.ndarray,
                                 query_points: np.ndarray,
                                 field_name: str,
                                 method_type: str = 'nearest') -> np.ndarray:
        """
        使用自定义方法插值单个场变量

        Args:
            vertices: 源网格顶点坐标
            values: 源网格上的场变量值
            query_points: 查询点坐标
            field_name: 场变量名称
            method_type: 插值方法类型 ('nearest' 或 'average')

        Returns:
            插值后的场变量值
        """
        from scipy.spatial import cKDTree

        # 构建KD树用于快速最近邻搜索
        tree = cKDTree(vertices)

        # 查找每个查询点的最近邻
        if method_type == 'nearest':
            # 方法1：使用最近的网格直接赋值
            distances, indices = tree.query(query_points, k=1)
            interpolated = values[indices]

            # 对于k=1，distances是1维数组
            distance_threshold = np.percentile(distances, 95)
            out_of_domain_mask = distances > distance_threshold * 2.0

        elif method_type == 'average':
            # 方法2：使用临界的3个网格值的平均值
            distances, indices = tree.query(query_points, k=3)

            if len(values.shape) == 1:
                # 标量场
                interpolated = np.mean(values[indices], axis=1)
            else:
                # 矢量场或张量场
                interpolated = np.mean(values[indices], axis=1)

            # 对于k>1，distances是2维数组
            distance_threshold = np.percentile(distances[:, 2], 95)
            out_of_domain_mask = distances[:, 0] > distance_threshold * 2.0
        else:
            raise ValueError(f"不支持的自定义插值方法: {method_type}")

        # 对于距离过远的点，设置为域外值

        interpolated[out_of_domain_mask] = self.out_of_domain_value

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