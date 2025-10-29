"""
STL文件读取模块

用于读取STL格式的表面几何文件，为SDF计算提供血管壁几何
根据CLAUDE.md需求：几何数据：Data/geo/portal_vein_A.stl，缩放比例0.001
"""

import numpy as np
import trimesh
from typing import Optional, Tuple, Dict, Any
import warnings
import os


class STLReader:
    """STL文件读取器"""

    def __init__(self):
        self.mesh = None
        self.vertices = None
        self.faces = None
        self.original_bounds = None
        self.scale_factor = 0.001  # CLAUDE.md指定的缩放比例

    def read_stl(self, file_path: str, scale_factor: float = 0.001) -> Optional[Dict[str, Any]]:
        """
        读取STL文件并应用缩放

        Args:
            file_path: STL文件路径
            scale_factor: 缩放比例，默认为0.001（根据CLAUDE.md）

        Returns:
            包含顶点和面片信息的字典，失败时返回None
        """
        try:
            print(f"Reading STL file: {file_path}")
            print(f"  - Scale factor: {scale_factor}")

            # 使用trimesh读取STL文件
            mesh = trimesh.load(file_path)

            if mesh is None:
                print(f"  [ERROR] Failed to load STL file")
                return None

            # 检查网格是否为三角形网格
            if not hasattr(mesh, 'faces') or mesh.faces is None:
                print(f"  [ERROR] No faces found in STL file")
                return None

            # 检查是否为三角形网格
            if len(mesh.faces.shape) != 2 or mesh.faces.shape[1] != 3:
                print(f"  [ERROR] STL file must contain triangular faces")
                print(f"  Found faces shape: {mesh.faces.shape}")
                return None

            # 检查网格是否为水密的
            if not mesh.is_watertight:
                print(f"  [WARNING] Mesh is not watertight, SDF calculation may be inaccurate")

            # 提取原始顶点和面片
            original_vertices = mesh.vertices.copy()
            faces = mesh.faces.copy()

            # 应用缩放
            vertices = original_vertices * scale_factor

            print(f"  [OK] Successfully loaded and scaled STL file")
            print(f"  - Original vertices: {len(original_vertices):,}")
            print(f"  - Faces (triangles): {len(faces):,}")
            print(f"  - Original bounds: [{np.min(original_vertices, axis=0)}, {np.max(original_vertices, axis=0)}]")
            print(f"  - Scaled bounds: [{np.min(vertices, axis=0)}, {np.max(vertices, axis=0)}]")
            print(f"  - Watertight: {mesh.is_watertight}")

            # 验证面片索引有效性
            max_vertex_idx = np.max(faces)
            if max_vertex_idx >= len(vertices):
                print(f"  [ERROR] Invalid face indices: max index {max_vertex_idx} >= num vertices {len(vertices)}")
                return None

            self.mesh = mesh
            self.vertices = vertices
            self.faces = faces
            self.original_bounds = (np.min(original_vertices, axis=0), np.max(original_vertices, axis=0))
            self.scale_factor = scale_factor

            return {
                'vertices': vertices,
                'faces': faces,
                'original_vertices': original_vertices,
                'scale_factor': scale_factor,
                'original_bounds': self.original_bounds,
                'scaled_bounds': (np.min(vertices, axis=0), np.max(vertices, axis=0)),
                'is_watertight': mesh.is_watertight,
                'num_vertices': len(vertices),
                'num_faces': len(faces)
            }

        except Exception as e:
            print(f"  [ERROR] Failed to read STL file: {e}")
            import traceback
            traceback.print_exc()
            return None

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
            'is_watertight': self.mesh.is_watertight if self.mesh else False,
            'scale_factor': self.scale_factor,
            'original_bounds': self.original_bounds,
            'scaled_bounds': None,
            'volume': None,
            'surface_area': None
        }

        if self.vertices is not None:
            bounds = (np.min(self.vertices, axis=0), np.max(self.vertices, axis=0))
            info['scaled_bounds'] = bounds
            info['extent'] = bounds[1] - bounds[0]

        if self.mesh is not None:
            try:
                info['volume'] = self.mesh.volume
                info['surface_area'] = self.mesh.area
            except:
                pass

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

        # 检查是否为三角形网格
        if self.faces is not None and len(self.faces.shape) == 2:
            if self.faces.shape[1] != 3:
                issues.append(f"Faces must be triangular, found shape {self.faces.shape}")

        # 检查是否为水密网格
        if not self.mesh.is_watertight:
            issues.append("Mesh is not watertight (has holes)")

        # 检查顶点数量
        if self.vertices is not None and len(self.vertices) < 3:
            issues.append("Mesh has fewer than 3 vertices")

        # 检查面片数量
        if self.faces is not None and len(self.faces) < 1:
            issues.append("Mesh has no faces")

        # 检查法向量方向
        try:
            if hasattr(self.mesh, 'face_normals'):
                normals = self.mesh.face_normals
                if np.any(np.isnan(normals)):
                    issues.append("Some face normals are NaN")
        except:
            issues.append("Could not compute face normals")

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

        # 更新mesh对象
        if self.mesh is not None:
            self.mesh.vertices = self.vertices

        print(f"  [OK] Applied additional scaling factor: {additional_scale}")
        print(f"  - New total scale factor: {self.scale_factor}")
        print(f"  - New bounds: [{np.min(self.vertices, axis=0)}, {np.max(self.vertices, axis=0)}]")


def load_portal_vein_geometry(base_dir: str = None) -> Optional[Dict[str, Any]]:
    """
    加载门静脉几何文件

    Args:
        base_dir: 项目根目录，如果为None则自动检测

    Returns:
        STL数据字典或None
    """
    if base_dir is None:
        # 更好的项目根目录检测方法
        current_file = os.path.abspath(__file__)
        # stl_reader.py 位于 src/ 目录中
        src_dir = os.path.dirname(current_file)
        base_dir = os.path.dirname(src_dir)

    stl_path = os.path.join(base_dir, "Data", "geo", "portal_vein_A.stl")

    # 调试输出路径
    print(f"Looking for STL file at: {stl_path}")
    print(f"File exists: {os.path.exists(stl_path)}")
    print(f"Base directory detected as: {base_dir}")

    if not os.path.exists(stl_path):
        print(f"[ERROR] STL file not found: {stl_path}")
        return None

    reader = STLReader()
    # 使用CLAUDE.md指定的缩放比例0.001
    result = reader.read_stl(stl_path, scale_factor=0.001)

    if result:
        print(f"[OK] Successfully loaded portal vein geometry")

        # 验证网格
        is_valid, issues = reader.validate_mesh()
        if not is_valid:
            print(f"[WARNING] Mesh validation issues: {issues}")

    return result


def test_stl_reader():
    """测试STL读取功能"""
    print("=== Portal Vein STL Reader Test ===\n")

    # 测试加载门静脉几何
    result = load_portal_vein_geometry()

    if result:
        print("✓ Successfully loaded portal vein geometry")
        print(f"  - Vertices: {result['num_vertices']:,}")
        print(f"  - Faces: {result['num_faces']:,}")
        print(f"  - Scale factor: {result['scale_factor']}")
        print(f"  - Original bounds: {result['original_bounds']}")
        print(f"  - Scaled bounds: {result['scaled_bounds']}")
        print(f"  - Watertight: {result['is_watertight']}")
    else:
        print("✗ Failed to load portal vein geometry")

    print("\nTest completed.")


if __name__ == "__main__":
    test_stl_reader()