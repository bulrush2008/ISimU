"""
VTP文件读取模块

用于读取VTK PolyData格式的几何文件，为SDF计算提供血管壁几何
支持VMR数据集的VTP格式几何文件
"""

import vtk
import numpy as np
from typing import Optional, Tuple, Dict, Any
import warnings
import os


class VTPReader:
    """VTP文件读取器"""

    def __init__(self):
        self.mesh = None
        self.vertices = None
        self.faces = None
        self.original_bounds = None
        self.scale_factor = 1.0  # VTP文件通常不需要缩放

    def read_vtp(self, file_path: str, scale_factor: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        读取VTP文件并应用缩放

        Args:
            file_path: VTP文件路径
            scale_factor: 缩放比例，默认为1.0

        Returns:
            包含顶点和面片信息的字典，失败时返回None
        """
        try:
            print(f"Reading VTP file: {file_path}")
            print(f"  - Scale factor: {scale_factor}")

            # 使用VTK读取VTP文件
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(file_path)
            reader.Update()

            polydata = reader.GetOutput()

            if polydata is None:
                print(f"  [ERROR] Failed to load VTP file")
                return None

            # 检查是否有几何数据
            if polydata.GetNumberOfPoints() == 0:
                print(f"  [ERROR] No points found in VTP file")
                return None

            if polydata.GetNumberOfPolys() == 0:
                print(f"  [ERROR] No polygons found in VTP file")
                return None

            # 提取顶点数据
            vertices = self._extract_vertices(polydata)

            # 提取面片数据
            faces = self._extract_triangular_faces(polydata)

            if len(faces) == 0:
                print(f"  [ERROR] No triangular faces found in VTP file")
                return None

            # 应用缩放
            if scale_factor != 1.0:
                vertices = vertices * scale_factor

            print(f"  [OK] Successfully loaded VTP file")
            print(f"  - Original vertices: {len(vertices):,}")
            print(f"  - Faces (triangles): {len(faces):,}")

            if scale_factor != 1.0:
                original_vertices = vertices / scale_factor
                print(f"  - Original bounds: [{np.min(original_vertices, axis=0)}, {np.max(original_vertices, axis=0)}]")
                print(f"  - Scaled bounds: [{np.min(vertices, axis=0)}, {np.max(vertices, axis=0)}]")
            else:
                print(f"  - Bounds: [{np.min(vertices, axis=0)}, {np.max(vertices, axis=0)}]")

            # 验证面片索引有效性
            max_vertex_idx = np.max(faces)
            if max_vertex_idx >= len(vertices):
                print(f"  [ERROR] Invalid face indices: max index {max_vertex_idx} >= num vertices {len(vertices)}")
                return None

            self.mesh = polydata
            self.vertices = vertices
            self.faces = faces
            self.original_bounds = (np.min(vertices, axis=0), np.max(vertices, axis=0))
            self.scale_factor = scale_factor

            return {
                'vertices': vertices,
                'faces': faces,
                'original_vertices': vertices / scale_factor if scale_factor != 1.0 else vertices,
                'scale_factor': scale_factor,
                'original_bounds': self.original_bounds,
                'scaled_bounds': (np.min(vertices, axis=0), np.max(vertices, axis=0)),
                'is_watertight': True,  # VTP通常是封闭的
                'num_vertices': len(vertices),
                'num_faces': len(faces),
                'format': 'VTP'
            }

        except Exception as e:
            print(f"  [ERROR] Failed to read VTP file: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_vertices(self, polydata: vtk.vtkPolyData) -> np.ndarray:
        """从VTK PolyData中提取顶点数据"""
        points = polydata.GetPoints()
        num_points = points.GetNumberOfPoints()

        vertices = np.zeros((num_points, 3))
        for i in range(num_points):
            vertices[i] = points.GetPoint(i)

        return vertices

    def _extract_triangular_faces(self, polydata: vtk.vtkPolyData) -> np.ndarray:
        """从VTK PolyData中提取三角形面片数据"""
        polys = polydata.GetPolys()
        num_cells = polys.GetNumberOfCells()

        faces = []
        polys.InitTraversal()
        idlist = vtk.vtkIdList()

        for i in range(num_cells):
            polys.GetNextCell(idlist)
            num_ids = idlist.GetNumberOfIds()

            # 只处理三角形
            if num_ids == 3:
                face_vertices = [idlist.GetId(j) for j in range(3)]
                faces.append(face_vertices)
            elif num_ids > 3:
                # 如果是多于三边形的多边形，进行三角化
                # 这里使用简单的扇形三角化方法
                for j in range(1, num_ids - 1):
                    face_vertices = [idlist.GetId(0), idlist.GetId(j), idlist.GetId(j + 1)]
                    faces.append(face_vertices)

        return np.array(faces, dtype=np.int32)

    def get_mesh_info(self) -> Dict[str, Any]:
        """
        获取网格信息

        Returns:
            网格详细信息
        """
        if self.mesh is None:
            return {'error': 'No mesh loaded'}

        info = {
            'num_vertices': len(self.vertices) if self.vertices is not None else 0,
            'num_faces': len(self.faces) if self.faces is not None else 0,
            'is_watertight': True,  # VTP通常是封闭的
            'scale_factor': self.scale_factor,
            'original_bounds': self.original_bounds,
            'scaled_bounds': None,
            'format': 'VTP'
        }

        if self.vertices is not None:
            bounds = (np.min(self.vertices, axis=0), np.max(self.vertices, axis=0))
            info['scaled_bounds'] = bounds
            info['extent'] = bounds[1] - bounds[0]

        return info

    def validate_mesh(self) -> Tuple[bool, list]:
        """
        验证网格质量

        Returns:
            (is_valid, issues_list)
        """
        if self.mesh is None:
            return False, ['No mesh loaded']

        issues = []

        # 检查是否有顶点数据
        if self.vertices is None or len(self.vertices) < 3:
            issues.append("Mesh has fewer than 3 vertices")

        # 检查是否有面片数据
        if self.faces is None or len(self.faces) < 1:
            issues.append("Mesh has no faces")

        # 检查面片是否为三角形
        if self.faces is not None and len(self.faces.shape) == 2:
            if self.faces.shape[1] != 3:
                issues.append(f"Faces must be triangular, found shape {self.faces.shape}")

        # 检查顶点数量
        if self.vertices is not None and len(self.vertices) < 3:
            issues.append("Mesh has fewer than 3 vertices")

        return len(issues) == 0, issues

    def apply_additional_scaling(self, additional_scale: float) -> None:
        """
        应用额外的缩放因子

        Args:
            additional_scale: 额外的缩放因子
        """
        if self.vertices is None:
            return

        self.vertices = self.vertices * additional_scale
        self.scale_factor *= additional_scale

        print(f"  [OK] Applied additional scaling factor: {additional_scale}")
        print(f"  - New total scale factor: {self.scale_factor}")
        print(f"  - New bounds: [{np.min(self.vertices, axis=0)}, {np.max(self.vertices, axis=0)}]")

    def export_to_stl(self, output_path: str) -> bool:
        """
        将读取的几何数据导出为STL文件用于验证

        Args:
            output_path: STL文件输出路径

        Returns:
            是否成功导出
        """
        if self.vertices is None or self.faces is None:
            print(f"[ERROR] No geometry data to export")
            return False

        try:
            print(f"Exporting geometry to STL: {output_path}")

            # 创建VTK PolyData对象
            points = vtk.vtkPoints()
            for vertex in self.vertices:
                points.InsertNextPoint(vertex[0], vertex[1], vertex[2])

            # 创建三角形
            triangles = vtk.vtkCellArray()
            for face in self.faces:
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, face[0])
                triangle.GetPointIds().SetId(1, face[1])
                triangle.GetPointIds().SetId(2, face[2])
                triangles.InsertNextCell(triangle)

            # 创建PolyData
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(triangles)

            # 写入STL文件
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(output_path)
            writer.SetInputData(polydata)
            writer.Write()

            print(f"[OK] STL file exported successfully: {output_path}")
            print(f"  - Vertices: {len(self.vertices):,}")
            print(f"  - Faces: {len(self.faces):,}")

            return True

        except Exception as e:
            print(f"[ERROR] Failed to export STL file: {e}")
            return False

    def export_to_vtp(self, output_path: str) -> bool:
        """
        将读取的几何数据重新导出为VTP文件用于验证

        Args:
            output_path: VTP文件输出路径

        Returns:
            是否成功导出
        """
        if self.vertices is None or self.faces is None:
            print(f"[ERROR] No geometry data to export")
            return False

        try:
            print(f"Exporting geometry to VTP: {output_path}")

            # 创建VTK PolyData对象（从读取的数据重建）
            points = vtk.vtkPoints()
            for vertex in self.vertices:
                points.InsertNextPoint(vertex[0], vertex[1], vertex[2])

            # 创建三角形
            triangles = vtk.vtkCellArray()
            for face in self.faces:
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, face[0])
                triangle.GetPointIds().SetId(1, face[1])
                triangle.GetPointIds().SetId(2, face[2])
                triangles.InsertNextCell(triangle)

            # 创建PolyData
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(triangles)

            # 写入VTP文件
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(output_path)
            writer.SetInputData(polydata)
            writer.Write()

            print(f"[OK] VTP file exported successfully: {output_path}")
            print(f"  - Vertices: {len(self.vertices):,}")
            print(f"  - Faces: {len(self.faces):,}")

            return True

        except Exception as e:
            print(f"[ERROR] Failed to export VTP file: {e}")
            return False


def load_vmr_geometry(case_id: str, base_dir: str = None) -> Optional[Dict[str, Any]]:
    """
    加载VMR数据集中的几何文件

    Args:
        case_id: VMR病例ID (如 "0007_H_AO_H")
        base_dir: 项目根目录，如果为None则自动检测

    Returns:
        VTP数据字典或None
    """
    if base_dir is None:
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file)
        base_dir = os.path.dirname(src_dir)

    # 加载配置文件
    config_file = os.path.join(base_dir, "data_VMR", "geo-flow.json")
    if not os.path.exists(config_file):
        print(f"[ERROR] VMR config file not found: {config_file}")
        return None

    try:
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)

        if case_id not in config:
            print(f"[ERROR] Case ID {case_id} not found in VMR config")
            return None

        vtp_path = os.path.join(base_dir, config[case_id]['geo'])

        print(f"Looking for VMR VTP file at: {vtp_path}")
        print(f"File exists: {os.path.exists(vtp_path)}")

        if not os.path.exists(vtp_path):
            print(f"[ERROR] VMR VTP file not found: {vtp_path}")
            return None

        reader = VTPReader()
        result = reader.read_vtp(vtp_path, scale_factor=1.0)  # VTP通常不需要缩放

        if result:
            print(f"[OK] Successfully loaded VMR geometry for case {case_id}")

            # 验证网格
            is_valid, issues = reader.validate_mesh()
            if not is_valid:
                print(f"[WARNING] VMR mesh validation issues: {issues}")

        return result

    except Exception as e:
        print(f"[ERROR] Failed to load VMR geometry: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_vtp_reader():
    """测试VTP读取功能"""
    print("=== VMR VTP Reader Test ===\n")

    # 测试加载第一个VMR算例
    result = load_vmr_geometry("0007_H_AO_H")

    if result:
        print("✓ Successfully loaded VMR geometry")
        print(f"  - Vertices: {result['num_vertices']:,}")
        print(f"  - Faces: {result['num_faces']:,}")
        print(f"  - Scale factor: {result['scale_factor']}")
        print(f"  - Format: {result['format']}")
        print(f"  - Bounds: {result['scaled_bounds']}")
        print(f"  - Watertight: {result['is_watertight']}")
    else:
        print("✗ Failed to load VMR geometry")

    print("\nTest completed.")


if __name__ == "__main__":
    test_vtp_reader()