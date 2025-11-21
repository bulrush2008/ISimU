"""
VMR数据加载器

支持从geo-flow.json配置文件加载VTP几何文件和VTU流场文件
提供统一的数据加载接口，支持单个算例和批量处理
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union
import json
import os
from pathlib import Path

from geometry_reader import GeometryReader
from data_reader import VTKReader
from sdf_utils_enhanced import EnhancedSDFCalculator


class VMRDataLoader:
    """
    VMR数据加载器

    功能：
    1. 从geo-flow.json加载配置
    2. 加载VTP几何文件
    3. 加载VTU流场文件
    4. 创建SDF计算器
    5. 支持单个算例和批量处理
    """

    def __init__(self, config_path: str = None):
        """
        初始化VMR数据加载器

        Args:
            config_path: 配置文件路径，默认使用data_VMR/geo-flow.json
        """
        if config_path is None:
            self.config_path = self._find_config_file()
        else:
            self.config_path = config_path

        self.config = None
        self.geometry_reader = GeometryReader()
        self.vtk_reader = VTKReader()
        self.sdf_calculator = EnhancedSDFCalculator()

        # 加载配置文件
        self._load_config()

    def _find_config_file(self) -> str:
        """查找配置文件"""
        # 尝试多个可能的路径
        possible_paths = [
            "data_VMR/geo-flow.json",
            "../data_VMR/geo-flow.json",
            os.path.join(os.getcwd(), "data_VMR", "geo-flow.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_VMR", "geo-flow.json")
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError("找不到VMR配置文件 geo-flow.json")

    def _load_config(self) -> None:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

            print(f"VMR配置文件加载成功: {self.config_path}")
            print(f"  - 可用算例: {list(self.config.keys())}")

        except Exception as e:
            print(f"VMR配置文件加载失败: {e}")
            raise

    def get_available_cases(self) -> List[str]:
        """获取可用的算例列表"""
        if self.config is None:
            return []
        return list(self.config.keys())

    def get_case_info(self, case_name: str) -> Dict[str, Any]:
        """
        获取指定算例的信息

        Args:
            case_name: 算例名称

        Returns:
            算例信息字典
        """
        if self.config is None:
            raise ValueError("配置文件未加载")

        if case_name not in self.config:
            raise ValueError(f"算例 '{case_name}' 不存在")

        case_config = self.config[case_name]
        geometry_file = case_config['geo']
        flow_file = case_config['flow']

        # 路径已经在配置文件中正确设置，直接使用
        # 根据运行目录调整路径
        current_dir = os.getcwd()
        if current_dir.endswith('src'):
            geometry_file = "../" + geometry_file
            flow_file = "../" + flow_file
        elif current_dir.endswith('examples'):
            geometry_file = "../" + geometry_file
            flow_file = "../" + flow_file

        info = {
            'case_name': case_name,
            'geometry_file': geometry_file,
            'flow_file': flow_file,
            'geometry_size': os.path.getsize(geometry_file) / (1024*1024) if os.path.exists(geometry_file) else 0,
            'flow_size': os.path.getsize(flow_file) / (1024*1024) if os.path.exists(flow_file) else 0,
            'geometry_exists': os.path.exists(geometry_file),
            'flow_exists': os.path.exists(flow_file)
        }

        return info

    def load_case(self, case_name: str,
                  fields: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        加载单个算例的完整数据

        Args:
            case_name: 算例名称
            fields: 需要加载的流场物理量，None表示加载全部

        Returns:
            完整的算例数据字典或None
        """
        print(f"加载VMR算例: {case_name}")

        # 获取算例配置
        case_info = self.get_case_info(case_name)

        if not case_info['geometry_exists']:
            print(f"  [ERROR] 几何文件不存在: {case_info['geometry_file']}")
            return None

        if not case_info['flow_exists']:
            print(f"  [ERROR] 流场文件不存在: {case_info['flow_file']}")
            return None

        result = {
            'case_name': case_name,
            'geometry_file': case_info['geometry_file'],
            'flow_file': case_info['flow_file'],
            'geometry_data': None,
            'flow_data': None,
            'sdf_calculator': None
        }

        try:
            # 1. 加载VTP几何数据
            print(f"  1. 加载几何数据...")
            geometry_data = self.geometry_reader.read_geometry(case_info['geometry_file'])

            if geometry_data is None:
                print(f"    [ERROR] 几何数据加载失败")
                return None

            result['geometry_data'] = geometry_data
            print(f"    [OK] 几何数据: {geometry_data['num_vertices']:,}顶点, {geometry_data['num_faces']:,}面片")

            # 2. 加载VTU流场数据
            print(f"  2. 加载流场数据...")
            flow_data = self.vtk_reader.read_vtu(case_info['flow_file'])

            if flow_data is None:
                print(f"    [ERROR] 流场数据加载失败")
                return None

            result['flow_data'] = flow_data

            # 过滤指定的物理场
            if fields is not None:
                filtered_flow_data = {}
                for field in fields:
                    if field in flow_data.get('point_data', {}):
                        filtered_flow_data[field] = flow_data['point_data'][field]
                    else:
                        print(f"    [WARNING] 物理场 '{field}' 不存在")

                flow_data['point_data'] = filtered_flow_data
                print(f"    [OK] 指定物理场: {list(filtered_flow_data.keys())}")
            else:
                print(f"    [OK] 全部物理场: {list(flow_data.get('point_data', {}).keys())}")

            # 3. 创建SDF计算器
            print(f"  3. 创建SDF计算器...")
            sdf = self.sdf_calculator.create_sdf_from_data(geometry_data)

            if sdf is None:
                print(f"    [ERROR] SDF计算器创建失败")
                return None

            result['sdf_calculator'] = sdf
            print(f"    [OK] SDF计算器创建成功")

            print(f"  [OK] 算例 '{case_name}' 加载完成")
            return result

        except Exception as e:
            print(f"  [ERROR] 算例加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_interpolator_data(self, case_name: str,
                               grid_size: tuple = (64, 64, 64),
                               fields: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        为插值器创建兼容的数据格式

        Args:
            case_name: 算例名称
            grid_size: 网格尺寸
            fields: 需要的物理场

        Returns:
            兼容插值器的数据格式或None
        """
        print(f"为插值器准备VMR数据: {case_name}")

        # 加载算例数据
        case_data = self.load_case(case_name, fields)

        if case_data is None:
            return None

        try:
            # 将VMR数据转换为插值器兼容格式
            # 使用流场数据作为主要数据源（包含几何和物理场）
            flow_data = case_data['flow_data']

            # 提取第一个非空的数据块（VTU只有一个块）
            if 'blocks' in flow_data and flow_data['blocks']:
                main_block = flow_data['blocks'][0]
            else:
                main_block = flow_data

            # 创建插值器兼容的数据格式
            interpolator_data = {
                'type': flow_data.get('type', 'UnstructuredGrid'),
                'num_points': main_block.get('num_points', 0),
                'num_cells': main_block.get('num_cells', 0),
                'vertices': main_block['vertices'],
                'point_data': main_block.get('point_data', {}),
                'cell_data': main_block.get('cell_data', {}),
                'geometry_data': case_data['geometry_data'],
                'sdf_calculator': case_data['sdf_calculator'],
                'case_name': case_name,
                'grid_size': grid_size,
                'bounds': None  # 将在插值器中设置
            }

            print(f"  [OK] 插值器数据准备完成")
            print(f"    - 数据类型: {interpolator_data['type']}")
            print(f"    - 顶点数: {interpolator_data['num_points']:,}")
            print(f"    - 单元数: {interpolator_data['num_cells']:,}")
            print(f"    - 物理场: {list(interpolator_data['point_data'].keys())}")
            print(f"    - SDF计算器: {'可用' if interpolator_data['sdf_calculator'] else '不可用'}")

            return interpolator_data

        except Exception as e:
            print(f"  [ERROR] 插值器数据准备失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def validate_case(self, case_name: str) -> Dict[str, Any]:
        """
        验证算例的完整性

        Args:
            case_name: 算例名称

        Returns:
            验证结果
        """
        result = {
            'case_name': case_name,
            'valid': False,
            'issues': [],
            'geometry_valid': False,
            'flow_valid': False,
            'sdf_valid': False
        }

        try:
            case_info = self.get_case_info(case_name)

            # 检查文件存在性
            if case_info['geometry_exists']:
                result['geometry_valid'] = True
            else:
                result['issues'].append(f"几何文件不存在: {case_info['geometry_file']}")

            if case_info['flow_exists']:
                result['flow_valid'] = True
            else:
                result['issues'].append(f"流场文件不存在: {case_info['flow_file']}")

            # 检查数据加载
            if result['geometry_valid'] and result['flow_valid']:
                case_data = self.load_case(case_name)

                if case_data:
                    result['sdf_valid'] = case_data['sdf_calculator'] is not None

                    if not result['sdf_valid']:
                        result['issues'].append("SDF计算器创建失败")
                else:
                    result['issues'].append("数据加载失败")

            # 总体有效性
            result['valid'] = (result['geometry_valid'] and
                              result['flow_valid'] and
                              result['sdf_valid'])

        except Exception as e:
            result['issues'].append(f"验证过程出错: {e}")

        return result

    def validate_all_cases(self) -> Dict[str, Any]:
        """
        验证所有算例

        Returns:
            验证结果汇总
        """
        print("验证所有VMR算例...")

        cases = self.get_available_cases()
        results = {}
        valid_count = 0

        for case_name in cases:
            print(f"\\n验证算例: {case_name}")
            result = self.validate_case(case_name)
            results[case_name] = result

            if result['valid']:
                valid_count += 1
                print(f"  [OK] 验证通过")
            else:
                print(f"  [ERROR] 验证失败:")
                for issue in result['issues']:
                    print(f"    * {issue}")

        print(f"\\n验证结果: {valid_count}/{len(cases)} 算例通过")

        return {
            'total_cases': len(cases),
            'valid_cases': valid_count,
            'invalid_cases': len(cases) - valid_count,
            'results': results
        }


def test_vmr_data_loader():
    """测试VMR数据加载器"""
    print("=== VMR数据加载器测试 ===")

    try:
        loader = VMRDataLoader()

        # 测试1: 获取可用算例
        print(f"\\n1. 获取可用算例:")
        cases = loader.get_available_cases()
        print(f"  可用算例: {cases}")

        if not cases:
            print(f"  [ERROR] 没有找到可用算例")
            return False

        # 测试2: 获取第一个算例信息
        first_case = cases[0]
        print(f"\\n2. 获取算例信息: {first_case}")
        case_info = loader.get_case_info(first_case)
        print(f"  - 几何文件: {case_info['geometry_file']}")
        print(f"  - 流场文件: {case_info['flow_file']}")
        print(f"  - 几何大小: {case_info['geometry_size']:.2f} MB")
        print(f"  - 流场大小: {case_info['flow_size']:.2f} MB")
        print(f"  - 文件存在: 几何={case_info['geometry_exists']}, 流场={case_info['flow_exists']}")

        # 测试3: 验证算例
        print(f"\\n3. 验证算例: {first_case}")
        validation = loader.validate_case(first_case)
        print(f"  - 验证结果: {'通过' if validation['valid'] else '失败'}")
        if validation['issues']:
            for issue in validation['issues']:
                print(f"    * {issue}")

        # 测试4: 加载算例数据（如果验证通过）
        if validation['valid']:
            print(f"\\n4. 加载算例数据: {first_case}")
            case_data = loader.load_case(first_case, ['pressure', 'velocity'])

            if case_data:
                print(f"  [OK] 算例数据加载成功")
                print(f"    - 几何数据: {case_data['geometry_data']['format']}")
                print(f"    - SDF计算器: {'可用' if case_data['sdf_calculator'] else '不可用'}")

                if case_data['flow_data']:
                    print(f"    - 流场物理场: {list(case_data['flow_data'].get('point_data', {}).keys())}")
            else:
                print(f"  [ERROR] 算例数据加载失败")

        print(f"\\n=== 测试完成 ===")
        return True

    except Exception as e:
        print(f"[ERROR] VMR数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_vmr_data_loader()