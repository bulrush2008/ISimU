"""
增强的SDF工具模块

支持多种几何源的SDF计算：STL文件、VTP文件、直接几何数据
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from typing import Tuple, Optional, List, Dict, Any, Union
import warnings
import os

from sdf_utils import VascularSDF  # 重用现有的VascularSDF类
from geometry_reader import GeometryReader


class EnhancedSDFCalculator:
    """
    增强的SDF计算器

    支持多种几何源：
    1. STL文件路径 (缩放0.001)
    2. VTP文件路径 (缩放1.0)
    3. 直接几何数据 (vertices, faces)
    4. JSON配置中的几何路径
    """

    def __init__(self):
        self.geometry_reader = GeometryReader()
        self.current_sdf = None
        self.current_geometry_info = None

    def create_sdf_from_file(self, geometry_path: str, scale_factor: float = None) -> Optional[VascularSDF]:
        """
        从几何文件创建SDF计算器

        Args:
            geometry_path: 几何文件路径 (STL或VTP)
            scale_factor: 缩放比例，None表示自动检测

        Returns:
            VascularSDF对象或None
        """
        try:
            print(f"Creating SDF from geometry file: {geometry_path}")

            # 使用统一几何读取器
            geometry_data = self.geometry_reader.read_geometry(geometry_path, scale_factor)

            if geometry_data is None:
                print(f"  [ERROR] Failed to load geometry file: {geometry_path}")
                return None

            return self.create_sdf_from_data(geometry_data)

        except Exception as e:
            print(f"  [ERROR] SDF creation failed: {e}")
            return None

    def create_sdf_from_data(self, geometry_data: Dict[str, Any]) -> Optional[VascularSDF]:
        """
        从几何数据创建SDF计算器

        Args:
            geometry_data: 几何数据字典，包含vertices和faces

        Returns:
            VascularSDF对象或None
        """
        try:
            vertices = geometry_data['vertices']
            faces = geometry_data['faces']

            if faces is None:
                print(f"  [ERROR] No faces found in geometry data")
                return None

            print(f"  Creating SDF from {geometry_data['format']} geometry...")
            print(f"    - Vertices: {len(vertices):,}")
            print(f"    - Faces: {len(faces):,}")
            print(f"    - Scale factor: {geometry_data['scale_factor']}")

            # 创建SDF计算器
            sdf = VascularSDF(vertices, faces)

            # 设置额外的几何信息
            sdf.geometry_format = geometry_data['format']
            sdf.geometry_source = geometry_data.get('file_path', 'direct_data')
            sdf.scale_factor = geometry_data['scale_factor']
            sdf.geometry_data = geometry_data

            # 保存当前状态
            self.current_sdf = sdf
            self.current_geometry_info = geometry_data

            print(f"  [OK] SDF calculator created successfully")
            return sdf

        except Exception as e:
            print(f"  [ERROR] SDF creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_sdf_from_vmr_config(self, case_name: str, config_path: str = "data_VMR/geo-flow.json") -> Optional[VascularSDF]:
        """
        从VMR配置创建SDF计算器

        Args:
            case_name: 算例名称 (如 "0007_H_AO_H")
            config_path: 配置文件路径

        Returns:
            VascularSDF对象或None
        """
        try:
            import json

            print(f"Creating SDF from VMR configuration: {case_name}")

            # 读取配置文件
            with open(config_path, 'r') as f:
                config = json.load(f)

            if case_name not in config:
                print(f"  [ERROR] Case '{case_name}' not found in configuration")
                return None

            case_config = config[case_name]
            geometry_path = case_config['geo']

            # 创建SDF（VTP格式，无需缩放）
            return self.create_sdf_from_file(geometry_path, scale_factor=1.0)

        except Exception as e:
            print(f"  [ERROR] VMR configuration processing failed: {e}")
            return None

    def get_current_sdf(self) -> Optional[VascularSDF]:
        """获取当前SDF计算器"""
        return self.current_sdf

    def get_current_geometry_info(self) -> Optional[Dict[str, Any]]:
        """获取当前几何信息"""
        return self.current_geometry_info

    def validate_sdf(self) -> Tuple[bool, List[str]]:
        """验证当前SDF计算器"""
        if self.current_sdf is None:
            return False, ["No SDF calculator created"]

        issues = []

        # 验证几何数据
        if self.current_geometry_info:
            vertices = self.current_geometry_info['vertices']
            faces = self.current_geometry_info['faces']

            if len(vertices) < 100:
                issues.append(f"Very few vertices: {len(vertices)}")

            if faces is None or len(faces) < 10:
                issues.append(f"Very few or no faces: {len(faces) if faces else 0}")

        return len(issues) == 0, issues


def create_sdf_enhanced(geometry_source: Union[str, Dict[str, Any]],
                       scale_factor: float = None,
                       source_type: str = 'auto') -> Optional[VascularSDF]:
    """
    增强的SDF创建函数，支持多种输入源

    Args:
        geometry_source: 几何源（文件路径、配置字典、几何数据字典）
        scale_factor: 缩放比例
        source_type: 源类型 ('auto', 'stl_file', 'vtp_file', 'vmr_config', 'direct_data')

    Returns:
        VascularSDF对象或None
    """
    calculator = EnhancedSDFCalculator()

    try:
        if source_type == 'auto':
            # 自动检测源类型
            if isinstance(geometry_source, str):
                if geometry_source.endswith('.stl'):
                    source_type = 'stl_file'
                elif geometry_source.endswith('.vtp'):
                    source_type = 'vtp_file'
                elif isinstance(geometry_source, str) and len(geometry_source.split()) == 1 and geometry_source.split()[0].isalnum():
                    source_type = 'vmr_config'  # 算例名称
                else:
                    source_type = 'stl_file'  # 默认
            elif isinstance(geometry_source, dict):
                if 'vertices' in geometry_source and 'faces' in geometry_source:
                    source_type = 'direct_data'
                elif 'geo' in geometry_source:
                    source_type = 'vmr_config'
                else:
                    source_type = 'direct_data'

        # 根据源类型创建SDF
        if source_type == 'stl_file':
            return calculator.create_sdf_from_file(geometry_source, scale_factor or 0.001)

        elif source_type == 'vtp_file':
            return calculator.create_sdf_from_file(geometry_source, scale_factor or 1.0)

        elif source_type == 'vmr_config':
            if isinstance(geometry_source, str):
                # 算例名称
                return calculator.create_sdf_from_vmr_config(geometry_source)
            elif isinstance(geometry_source, dict):
                # 配置字典
                case_name = list(geometry_source.keys())[0]
                return calculator.create_sdf_from_vmr_config(case_name)

        elif source_type == 'direct_data':
            return calculator.create_sdf_from_data(geometry_source)

        else:
            print(f"  [ERROR] Unknown source type: {source_type}")
            return None

    except Exception as e:
        print(f"  [ERROR] Enhanced SDF creation failed: {e}")
        return None


# 向后兼容的函数
def create_sdf_from_vtk_data_enhanced(vtk_data: Dict[str, Any],
                                    geometry_source: str = None) -> Optional[VascularSDF]:
    """
    向后兼容的SDF创建函数，支持指定几何源

    Args:
        vtk_data: VTK数据（保持向后兼容）
        geometry_source: 几何源路径，None表示使用默认STL

    Returns:
        VascularSDF对象或None
    """
    if geometry_source is None:
        # 使用原有逻辑，尝试STL文件
        from sdf_utils import create_sdf_from_vtk_data
        return create_sdf_from_vtk_data(vtk_data)
    else:
        # 使用新的增强逻辑
        return create_sdf_enhanced(geometry_source)


def test_enhanced_sdf():
    """测试增强的SDF计算器"""
    print("=== 增强SDF计算器测试 ===")

    calculator = EnhancedSDFCalculator()

    # 测试1: STL文件（如果存在）
    stl_file = "data_UNS/geo/portal_vein_A.stl"
    if os.path.exists(stl_file):
        print(f"\\n1. 测试STL文件: {stl_file}")
        sdf = calculator.create_sdf_from_file(stl_file)
        if sdf:
            print(f"  [OK] STL SDF创建成功")
            info = calculator.get_current_geometry_info()
            print(f"    格式: {info['format']}, 缩放: {info['scale_factor']}")

    # 测试2: VTP文件（如果存在）
    vtp_file = "data_VMR/0007_H_AO_H/Simulations/0090_0001/check/initial.vtp"
    if os.path.exists(vtp_file):
        print(f"\\n2. 测试VTP文件: {vtp_file}")
        sdf = calculator.create_sdf_from_file(vtp_file)
        if sdf:
            print(f"  [OK] VTP SDF创建成功")
            info = calculator.get_current_geometry_info()
            print(f"    格式: {info['format']}, 缩放: {info['scale_factor']}")

    # 测试3: VMR配置
    if os.path.exists("data_VMR/geo-flow.json"):
        print(f"\\n3. 测试VMR配置: 0007_H_AO_H")
        sdf = calculator.create_sdf_from_vmr_config("0007_H_AO_H")
        if sdf:
            print(f"  [OK] VMR SDF创建成功")
            info = calculator.get_current_geometry_info()
            print(f"    格式: {info['format']}, 缩放: {info['scale_factor']}")

    print(f"\\n=== 测试完成 ===")


if __name__ == "__main__":
    import os
    test_enhanced_sdf()