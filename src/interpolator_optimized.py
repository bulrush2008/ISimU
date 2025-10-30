"""
优化版本的插值器，修复SDF重复计算问题
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial import Delaunay
from typing import Dict, List, Any, Tuple, Optional
import warnings

from data_reader import VTKReader
from sdf_utils import create_sdf_from_vtk_data


class OptimizedGridInterpolator:
    """优化的网格插值器，避免重复SDF计算"""

    def __init__(self,
                 grid_size: Tuple[int, int, int] = (32, 32, 32),
                 method: str = 'linear',
                 out_of_domain_value: float = 0.0,
                 use_sdf: bool = True,
                 batch_size: int = 15000):  # 增大批处理大小
        """
        初始化优化插值器

        Args:
            grid_size: 网格尺寸 (nx, ny, nz)
            method: 插值方法 ('linear', 'nearest')
            out_of_domain_value: 域外点的赋值
            use_sdf: 是否使用SDF判断域内外
            batch_size: SDF计算的批处理大小
        """
        self.grid_size = grid_size
        self.method = method
        self.out_of_domain_value = out_of_domain_value
        self.use_sdf = use_sdf
        self.batch_size = batch_size

        self.cartesian_grid = None
        self.sdf_calculator = None
        self.sdf_values = None  # 缓存SDF值
        self.inside_mask = None  # 缓存内部掩码
        self.outside_mask = None  # 缓存外部掩码

    def setup_cartesian_grid(self, vertices: np.ndarray):
        """设置笛卡尔网格"""
        print(f"Cartesian grid setup completed:")

        # 计算顶点边界
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)

        # 稍微扩展边界
        margin = 0.01 * (max_coords - min_coords)
        min_coords -= margin
        max_coords += margin

        # 创建网格
        x = np.linspace(min_coords[0], max_coords[0], self.grid_size[0])
        y = np.linspace(min_coords[1], max_coords[1], self.grid_size[1])
        z = np.linspace(min_coords[2], max_coords[2], self.grid_size[2])

        self.cartesian_grid = np.meshgrid(x, y, z, indexing='ij')

        print(f"  - Grid size: {self.grid_size}")
        print(f"  - X range: [{min_coords[0]:.3f}, {max_coords[0]:.3f}]")
        print(f"  - Y range: [{min_coords[1]:.3f}, {max_coords[1]:.3f}]")
        print(f"  - Z range: [{min_coords[2]:.3f}, {max_coords[2]:.3f}]")

    def compute_sdf_once(self, query_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        只计算一次SDF，并缓存结果

        Returns:
            (sdf_values, inside_mask, outside_mask)
        """
        if self.sdf_values is None:
            # 首次计算
            print("  Computing SDF values for all grid points (optimized batch size)...")

            # 使用更大的批处理大小
            if hasattr(self.sdf_calculator, 'batch_size'):
                original_batch_size = self.sdf_calculator.batch_size
                self.sdf_calculator.batch_size = self.batch_size

            self.sdf_values = self.sdf_calculator.compute_sdf(query_points)

            # 恢复原始批处理大小
            if hasattr(self.sdf_calculator, 'batch_size'):
                self.sdf_calculator.batch_size = original_batch_size

            # 计算内外掩码（基于已计算的SDF值，不重复计算）
            self.inside_mask = self.sdf_values > 0.0
            self.outside_mask = ~self.inside_mask

            inside_count = np.sum(self.inside_mask)
            outside_count = np.sum(self.outside_mask)
            total_count = len(query_points)
            print(f"  - Inside vessel: {inside_count:,} ({inside_count/total_count*100:.1f}%)")
            print(f"  - Outside vessel: {outside_count:,} ({outside_count/total_count*100:.1f}%)")
            print(f"  - SDF range: [{np.min(self.sdf_values):.3e}, {np.max(self.sdf_values):.3e}]")

        return self.sdf_values, self.inside_mask, self.outside_mask

    def interpolate(self, vtk_data: Dict[str, Any], fields: List[str] = None) -> Dict[str, Any]:
        """
        执行优化的插值计算
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

        # 创建SDF计算器（如果需要）
        if self.use_sdf:
            print("Creating SDF calculator...")
            self.sdf_calculator = create_sdf_from_vtk_data(vtk_data)

            if self.sdf_calculator is not None:
                print("  [OK] SDF calculator created successfully")

                # 只计算一次SDF
                sdf_values, inside_mask, outside_mask = self.compute_sdf_once(query_points)
            else:
                print("  [WARNING] Failed to create SDF calculator, using all points")
                inside_mask = np.ones(len(query_points), dtype=bool)
                outside_mask = np.zeros(len(query_points), dtype=bool)
                sdf_values = np.ones(len(query_points)) * 1.0
        else:
            print("SDF disabled, using all points for interpolation")
            inside_mask = np.ones(len(query_points), dtype=bool)
            outside_mask = np.zeros(len(query_points), dtype=bool)
            sdf_values = np.ones(len(query_points)) * 1.0

        # 确定要插值的场
        if fields is None:
            fields = list(point_data.keys())

        print(f"Starting interpolation, fields: {fields}")

        # 计算网格边界
        x_coords = self.cartesian_grid[0]
        y_coords = self.cartesian_grid[1]
        z_coords = self.cartesian_grid[2]

        bounds = np.array([
            [float(np.min(x_coords)), float(np.max(x_coords))],
            [float(np.min(y_coords)), float(np.max(y_coords))],
            [float(np.min(z_coords)), float(np.max(z_coords))]
        ])

        result = {
            'grid': {
                'x': x_coords,
                'y': y_coords,
                'z': z_coords
            },
            'grid_coordinates': {  # hdf5_storage需要的字段
                'x': x_coords,
                'y': y_coords,
                'z': z_coords
            },
            'bounds': bounds,  # 添加缺失的bounds
            'fields': {},
            'sdf_used': self.use_sdf and self.sdf_calculator is not None,
            'grid_size': self.grid_size
        }

        # 保存SDF字段
        if self.use_sdf and sdf_values is not None:
            result['fields']['SDF'] = sdf_values.reshape(self.grid_size)
            print(f"  [OK] SDF: ({sdf_values.size}) -> {self.grid_size}")
            print(f"    - Positive (inside): {np.sum(inside_mask)} ({np.sum(inside_mask)/len(inside_mask)*100:.1f}%)")
            print(f"    - Negative (outside): {np.sum(outside_mask)} ({np.sum(outside_mask)/len(outside_mask)*100:.1f}%)")

        # 调试：检查result字典
        print(f"Debug: result keys = {list(result.keys())}")
        print(f"Debug: grid_size = {result.get('grid_size', 'NOT FOUND')}")

        # 对每个场进行插值
        for field_name in fields:
            if field_name in point_data:
                print(f"  Processing {field_name} from point_data")
                field_values = point_data[field_name]

                # 检查维度
                if len(field_values.shape) == 1:
                    # 标量场
                    interpolated = self._interpolate_scalar_field_optimized(
                        vertices, field_values, query_points,
                        inside_mask, outside_mask
                    )
                elif len(field_values.shape) == 2 and field_values.shape[1] in [2, 3]:
                    # 矢量场
                    interpolated = self._interpolate_vector_field_optimized(
                        vertices, field_values, query_points,
                        inside_mask, outside_mask
                    )
                else:
                    print(f"    [WARNING] Skipping field {field_name} with unsupported shape {field_values.shape}")
                    continue

                result['fields'][field_name] = interpolated.reshape(
                    self.grid_size + interpolated.shape[1:]
                )
                print(f"  [OK] {field_name}: {field_values.shape} -> {self.grid_size + interpolated.shape[1:]}")
            else:
                print(f"    [WARNING] Field {field_name} not found in point_data")

        return result

    def _interpolate_scalar_field_optimized(self, vertices: np.ndarray, values: np.ndarray,
                                          query_points: np.ndarray, inside_mask: np.ndarray,
                                          outside_mask: np.ndarray) -> np.ndarray:
        """优化的标量场插值"""
        # 使用LinearNDInterpolator优化性能
        try:
            interpolator = LinearNDInterpolator(vertices, values, fill_value=0.0)
            interpolated_values = interpolator(query_points)
        except Exception as e:
            print(f"    LinearNDInterpolator failed: {e}, falling back to griddata")
            interpolated_values = griddata(vertices, values, query_points,
                                         method=self.method, fill_value=0.0)

        # 应用SDF掩码
        interpolated_values[~inside_mask] = self.out_of_domain_value
        return interpolated_values

    def _interpolate_vector_field_optimized(self, vertices: np.ndarray, values: np.ndarray,
                                          query_points: np.ndarray, inside_mask: np.ndarray,
                                          outside_mask: np.ndarray) -> np.ndarray:
        """优化的矢量场插值"""
        num_components = values.shape[1]
        interpolated = np.zeros((len(query_points), num_components))

        # 对每个分量分别插值
        for i in range(num_components):
            try:
                interpolator = LinearNDInterpolator(vertices, values[:, i], fill_value=0.0)
                interpolated[:, i] = interpolator(query_points)
            except Exception as e:
                print(f"    LinearNDInterpolator failed for component {i}: {e}")
                interpolated[:, i] = griddata(vertices, values[:, i], query_points,
                                            method=self.method, fill_value=0.0)

        # 应用SDF掩码
        interpolated[~inside_mask] = self.out_of_domain_value
        return interpolated