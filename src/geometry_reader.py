"""
统一几何文件读取器

支持STL和VTP两种格式的几何文件读取，为SDF计算提供统一的接口
"""

import numpy as np
from typing import Optional, Dict, Any, Union, Tuple, List
import warnings
import os


class GeometryReader:
    """
    统一几何文件读取器

    支持格式：
    - STL: 需要trimesh库处理，支持0.001缩放
    - VTP: 直接使用VTK库处理，无需缩放(比例=1.0)
    """

    def __init__(self):
        self.stl_reader = None
        self.vtp_reader = None
        self.last_loaded_type = None
        self.last_loaded_data = None

    def read_geometry(self, file_path: str, scale_factor: float = None) -> Optional[Dict[str, Any]]:
        """
        统一几何文件读取接口

        Args:
            file_path: 几何文件路径 (STL或VTP)
            scale_factor: 缩放比例，None表示自动检测

        Returns:
            几何数据字典或None
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"几何文件不存在: {file_path}")

        # 根据文件扩展名确定格式
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.stl':
            return self._read_stl(file_path, scale_factor or 0.001)
        elif file_ext == '.vtp':
            return self._read_vtp(file_path, scale_factor or 1.0)
        else:
            raise ValueError(f"不支持的几何文件格式: {file_ext}。支持格式: .stl, .vtp")

    def _read_stl(self, file_path: str, scale_factor: float) -> Optional[Dict[str, Any]]:
        """读取STL格式几何文件"""
        try:
            # 延迟导入STL读取器
            from stl_reader import STLReader

            if self.stl_reader is None:
                self.stl_reader = STLReader()

            print(f"Reading STL geometry: {file_path}")
            print(f"  - Scale factor: {scale_factor}")

            data = self.stl_reader.read_stl(file_path, scale_factor)

            if data:
                # 统一数据格式
                self.last_loaded_type = 'stl'
                self.last_loaded_data = {
                    'vertices': data['vertices'],
                    'faces': data['faces'],
                    'scale_factor': data['scale_factor'],
                    'format': 'stl',
                    'file_path': file_path,
                    'num_vertices': data['num_vertices'],
                    'num_faces': data['num_faces'],
                    'is_watertight': data.get('is_watertight', False),
                    'original_bounds': data.get('original_bounds'),
                    'scaled_bounds': data.get('scaled_bounds')
                }
                print(f"  [OK] STL geometry loaded successfully")
                return self.last_loaded_data

        except Exception as e:
            print(f"[ERROR] STL文件读取失败: {e}")
            return None

    def _read_vtp(self, file_path: str, scale_factor: float) -> Optional[Dict[str, Any]]:
        """读取VTP格式几何文件"""
        try:
            import vtk
            from vtk.util import numpy_support

            print(f"Reading VTP geometry: {file_path}")
            print(f"  - Scale factor: {scale_factor}")

            # 使用VTK读取VTP文件
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(file_path)
            reader.Update()

            poly_data = reader.GetOutput()

            if poly_data.GetNumberOfPoints() == 0:
                print(f"  [WARNING] VTP文件包含0个点")
                return None

            # 提取顶点
            points = poly_data.GetPoints()
            vertices = numpy_support.vtk_to_numpy(points.GetData())

            # 应用缩放
            if scale_factor != 1.0:
                vertices = vertices * scale_factor

            # 提取面片
            faces = None
            if poly_data.GetNumberOfPolys() > 0:
                polys = poly_data.GetPolys()
                faces = self._extract_faces_from_vtk(polys, vertices.shape[0])

            # 统一数据格式
            self.last_loaded_type = 'vtp'
            self.last_loaded_data = {
                'vertices': vertices,
                'faces': faces,
                'scale_factor': scale_factor,
                'format': 'vtp',
                'file_path': file_path,
                'num_vertices': len(vertices),
                'num_faces': len(faces) if faces is not None else 0,
                'is_watertight': None,  # VTP不提供此信息
                'original_bounds': (vertices.min(axis=0), vertices.max(axis=0)),
                'scaled_bounds': (vertices.min(axis=0), vertices.max(axis=0))
            }

            print(f"  [OK] VTP geometry loaded successfully")
            print(f"    - Vertices: {len(vertices):,}")
            print(f"    - Faces: {len(faces) if faces is not None else 0:,}")
            print(f"    - Scale factor: {scale_factor}")

            return self.last_loaded_data

        except Exception as e:
            print(f"[ERROR] VTP文件读取失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_faces_from_vtk(self, polys, num_vertices):
        """从VTK多边形数据中提取面片索引"""
        try:
            import vtk
            from vtk.util import numpy_support

            # 转换VTK多边形数据
            polys_data = numpy_support.vtk_to_numpy(polys.GetData())

            if len(polys_data) == 0:
                return None

            # 解析多边形数据
            # VTK格式: [n1, i1, i2, ..., i_n1, n2, j1, j2, ..., j_n2, ...]
            faces = []
            idx = 0

            while idx < len(polys_data):
                n_verts = polys_data[idx]
                if n_verts < 3:
                    idx += 1
                    continue

                face_indices = polys_data[idx+1:idx+1+n_verts]

                # 只保留三角形和四边形
                if n_verts in [3, 4]:
                    # 确保索引有效
                    if np.all(face_indices < num_vertices) and np.all(face_indices >= 0):
                        faces.append(face_indices)

                idx += n_verts + 1

            if faces:
                faces_array = np.array(faces)
                print(f"    - Extracted {len(faces)} polygons (triangles/quads)")
                return faces_array
            else:
                print(f"    [WARNING] No valid polygons found")
                return None

        except Exception as e:
            print(f"  [ERROR] 面片提取失败: {e}")
            return None

    def get_geometry_info(self) -> Dict[str, Any]:
        """获取最后加载的几何信息"""
        if self.last_loaded_data is None:
            return {'error': 'No geometry loaded'}

        info = {
            'format': self.last_loaded_data['format'],
            'file_path': self.last_loaded_data['file_path'],
            'num_vertices': self.last_loaded_data['num_vertices'],
            'num_faces': self.last_loaded_data['num_faces'],
            'scale_factor': self.last_loaded_data['scale_factor'],
            'bounds': self.last_loaded_data['scaled_bounds']
        }

        if 'is_watertight' in self.last_loaded_data:
            info['is_watertight'] = self.last_loaded_data['is_watertight']

        return info

    def validate_for_sdf(self) -> Tuple[bool, List[str]]:
        """验证几何数据是否适合SDF计算"""
        if self.last_loaded_data is None:
            return False, ['No geometry loaded']

        issues = []

        # 检查顶点数量
        if self.last_loaded_data['num_vertices'] < 3:
            issues.append("Geometry has fewer than 3 vertices")

        # 检查面片数量
        if self.last_loaded_data['num_faces'] < 1:
            issues.append("Geometry has no faces")

        # 检查缩放因子
        scale_factor = self.last_loaded_data['scale_factor']
        if scale_factor <= 0:
            issues.append(f"Invalid scale factor: {scale_factor}")

        # 检查VTP特有问题
        if self.last_loaded_type == 'vtp':
            if self.last_loaded_data['faces'] is None:
                issues.append("VTP geometry: no faces extracted for SDF calculation")

        return len(issues) == 0, issues


def test_geometry_reader():
    """测试几何读取器"""
    print("=== 几何读取器测试 ===")

    reader = GeometryReader()

    # 测试STL读取（如果文件存在）
    stl_file = "data_UNS/geo/portal_vein_A.stl"
    if os.path.exists(stl_file):
        print(f"\\n1. 测试STL读取: {stl_file}")
        try:
            stl_data = reader.read_geometry(stl_file)
            if stl_data:
                print(f"  [OK] STL读取成功")
                info = reader.get_geometry_info()
                print(f"    - 格式: {info['format']}")
                print(f"    - 顶点数: {info['num_vertices']:,}")
                print(f"    - 面片数: {info['num_faces']:,}")
                print(f"    - 缩放: {info['scale_factor']}")
        except Exception as e:
            print(f"  [ERROR] STL读取失败: {e}")

    # 测试VTP读取（如果文件存在）
    vtp_file = "data_VMR/0007_H_AO_H/Simulations/0090_0001/check/initial.vtp"
    if os.path.exists(vtp_file):
        print(f"\\n2. 测试VTP读取: {vtp_file}")
        try:
            vtp_data = reader.read_geometry(vtp_file)
            if vtp_data:
                print(f"  [OK] VTP读取成功")
                info = reader.get_geometry_info()
                print(f"    - 格式: {info['format']}")
                print(f"    - 顶点数: {info['num_vertices']:,}")
                print(f"    - 面片数: {info['num_faces']:,}")
                print(f"    - 缩放: {info['scale_factor']}")
        except Exception as e:
            print(f"  [ERROR] VTP读取失败: {e}")

    print(f"\\n=== 测试完成 ===")


if __name__ == "__main__":
    test_geometry_reader()