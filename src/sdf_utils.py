"""
符号距离场（SDF）工具模块

用于判断笛卡尔网格点是否在血管内部
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from typing import Tuple, Optional, List, Dict, Any
import warnings


class VascularSDF:
    """
    血管符号距离场计算器

    使用血管表面网格计算符号距离场：
    - phi > 0: 血管内部
    - phi < 0: 血管外部
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        初始化SDF计算器

        Args:
            vertices: 顶点坐标 (N, 3)
            faces: 面片索引 (M, 3)
        """
        self.vertices = vertices
        self.faces = faces

        # 创建trimesh对象
        try:
            self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if not self.mesh.is_watertight:
                warnings.warn("网格不是封闭的，SDF计算可能不准确")
        except Exception as e:
            warnings.warn(f"创建trimesh对象失败: {e}")
            self.mesh = None

        # 构建KD树用于快速最近邻搜索
        self.kdtree = cKDTree(vertices)

        # 预计算面片法向量
        self.face_normals = self._compute_face_normals()

    def _compute_face_normals(self) -> np.ndarray:
        """计算面片法向量"""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        # 计算两个边向量
        edge1 = v1 - v0
        edge2 = v2 - v0

        # 叉积得到法向量
        normals = np.cross(edge1, edge2)

        # 归一化
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-12)

        return normals

    def compute_sdf(self, points: np.ndarray) -> np.ndarray:
        """
        计算一组点的符号距离场值

        Args:
            points: 查询点坐标 (N, 3)

        Returns:
            符号距离场值 (N,)
        """
        if self.mesh is not None:
            return self._compute_sdf_trimesh(points)
        else:
            return self._compute_sdf_approximate(points)

    def _compute_sdf_trimesh(self, points: np.ndarray) -> np.ndarray:
        """使用trimesh计算精确SDF"""
        try:
            # 计算到表面的距离
            distances = self.mesh.nearest.signed_distance(points)
            return distances
        except Exception as e:
            warnings.warn(f"trimesh SDF计算失败，使用近似方法: {e}")
            return self._compute_sdf_approximate(points)

    def _compute_sdf_approximate(self, points: np.ndarray) -> np.ndarray:
        """近似SDF计算（当trimesh不可用时）"""
        # 找到最近的面片
        distances, indices = self.kdtree.query(points, k=1)

        # 计算符号
        # 这里使用简化的方法：基于距离和法向量方向
        sdf_values = np.zeros(len(points))

        for i, (point, dist, idx) in enumerate(zip(points, distances, indices)):
            # 找到包含最近顶点的面片
            nearby_faces = np.where(np.any(self.faces == idx, axis=1))[0]

            if len(nearby_faces) > 0:
                # 使用第一个邻近面片的法向量
                face_idx = nearby_faces[0]
                face_normal = self.face_normals[face_idx]
                face_center = np.mean(self.vertices[self.faces[face_idx]], axis=0)

                # 计算向量
                to_point = point - face_center

                # 符号基于法向量方向
                sign = np.sign(np.dot(to_point, face_normal))

                # 距离始终为正
                sdf_values[i] = sign * dist
            else:
                # 如果找不到邻近面片，使用距离作为正值
                sdf_values[i] = dist

        return sdf_values

    def is_inside_vessel(self, points: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        """
        判断点是否在血管内部

        Args:
            points: 查询点坐标 (N, 3)
            tolerance: 容忍度，默认为0

        Returns:
            布尔数组 (N,)，True表示在血管内部
        """
        sdf_values = self.compute_sdf(points)
        return sdf_values > tolerance

    def get_inside_outside_mask(self, points: np.ndarray, tolerance: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取血管内外的掩码

        Args:
            points: 查询点坐标 (N, 3)
            tolerance: 容忍度

        Returns:
            (inside_mask, outside_mask): 两个布尔数组
        """
        sdf_values = self.compute_sdf(points)
        inside_mask = sdf_values > tolerance
        outside_mask = ~inside_mask
        return inside_mask, outside_mask


def extract_surface_from_vtk_data(vtk_data: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    从VTK数据中提取表面网格

    Args:
        vtk_data: VTK读取的数据

    Returns:
        (vertices, faces) 或 None
    """
    # 寻找包含表面数据的块
    surface_block = None

    for block in vtk_data['blocks']:
        if block['num_points'] > 0 and 'faces' in block:
            surface_block = block
            break

    if surface_block is None:
        warnings.warn("未找到表面数据，尝试从体数据生成表面")
        return None

    vertices = surface_block['vertices']
    faces = surface_block.get('faces')

    if faces is None:
        warnings.warn("未找到面片数据")
        return None

    return vertices, faces


def create_sdf_from_vtk_data(vtk_data: Dict[str, Any]) -> Optional[VascularSDF]:
    """
    从VTK数据创建SDF计算器

    Args:
        vtk_data: VTK读取的数据

    Returns:
        VascularSDF对象或None
    """
    surface_data = extract_surface_from_vtk_data(vtk_data)

    if surface_data is None:
        return None

    vertices, faces = surface_data
    return VascularSDF(vertices, faces)