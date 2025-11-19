"""
VTK文件读取模块

用于读取UNS求解器输出的VTK格式文件
"""

import vtk
import numpy as np
from typing import Dict, Any, Optional, List


class VTKReader:
    """VTK文件读取器"""

    def __init__(self):
        self.reader = None
        self.data = None

    def read_vtm(self, file_path: str) -> Dict[str, Any]:
        """
        读取VTM多块数据集文件

        Args:
            file_path: VTM文件路径

        Returns:
            包含网格数据和物理场的字典
        """
        # 创建多块数据集读取器
        reader = vtk.vtkXMLMultiBlockDataReader()
        reader.SetFileName(file_path)
        reader.Update()

        multi_block = reader.GetOutput()

        return self._extract_multi_block_data(multi_block)

    def read_vtu(self, file_path: str) -> Dict[str, Any]:
        """
        读取VTU非结构网格文件

        Args:
            file_path: VTU文件路径

        Returns:
            包含网格数据和物理场的字典
        """
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(file_path)
        reader.Update()

        unstructured_grid = reader.GetOutput()

        return self._extract_unstructured_grid_data(unstructured_grid)

    def _extract_multi_block_data(self, multi_block: vtk.vtkMultiBlockDataSet) -> Dict[str, Any]:
        """从多块数据集中提取数据"""
        result = {
            'blocks': [],
            'num_blocks': multi_block.GetNumberOfBlocks()
        }

        for i in range(multi_block.GetNumberOfBlocks()):
            block = multi_block.GetBlock(i)
            if block:
                block_data = self._extract_block_data(block, i)
                result['blocks'].append(block_data)

        return result

    def _extract_block_data(self, block, block_index: int) -> Dict[str, Any]:
        """提取单个块的数据"""
        if block.IsA('vtkUnstructuredGrid'):
            return self._extract_unstructured_grid_data(block, block_index)
        elif block.IsA('vtkPolyData'):
            return self._extract_polydata(block, block_index)
        else:
            return {'type': block.GetClassName(), 'data': None}

    def _extract_unstructured_grid_data(self, grid, block_index: int = 0) -> Dict[str, Any]:
        """提取非结构网格数据"""
        # 获取点坐标
        points = grid.GetPoints()
        if points:
            num_points = points.GetNumberOfPoints()
            vertices = np.zeros((num_points, 3))
            for i in range(num_points):
                vertices[i] = points.GetPoint(i)
        else:
            vertices = np.array([])

        # 获取单元信息和面片
        cells = grid.GetCells()
        num_cells = grid.GetNumberOfCells()
        faces = self._extract_faces_from_unstructured_grid(grid)

        # 获取点数据和单元数据
        point_data = {}
        cell_data = {}

        # 提取点数据（物理场变量）
        point_data_vtk = grid.GetPointData()
        if point_data_vtk:
            for i in range(point_data_vtk.GetNumberOfArrays()):
                array = point_data_vtk.GetArray(i)
                array_name = array.GetName()
                data = self._vtk_array_to_numpy(array)
                point_data[array_name] = data

        # 提取单元数据
        cell_data_vtk = grid.GetCellData()
        if cell_data_vtk:
            for i in range(cell_data_vtk.GetNumberOfArrays()):
                array = cell_data_vtk.GetArray(i)
                array_name = array.GetName()
                data = self._vtk_array_to_numpy(array)
                cell_data[array_name] = data

        return {
            'type': 'UnstructuredGrid',
            'block_index': block_index,
            'vertices': vertices,
            'faces': faces,
            'num_points': vertices.shape[0] if vertices.size > 0 else 0,
            'num_cells': num_cells,
            'point_data': point_data,
            'cell_data': cell_data
        }

    def _extract_polydata(self, polydata, block_index: int = 0) -> Dict[str, Any]:
        """提取多边形数据"""
        # 获取点坐标
        points = polydata.GetPoints()
        if points:
            num_points = points.GetNumberOfPoints()
            vertices = np.zeros((num_points, 3))
            for i in range(num_points):
                vertices[i] = points.GetPoint(i)
        else:
            vertices = np.array([])

        # 获取点数据
        point_data = {}
        point_data_vtk = polydata.GetPointData()
        if point_data_vtk:
            for i in range(point_data_vtk.GetNumberOfArrays()):
                array = point_data_vtk.GetArray(i)
                array_name = array.GetName()
                data = self._vtk_array_to_numpy(array)
                point_data[array_name] = data

        return {
            'type': 'PolyData',
            'block_index': block_index,
            'vertices': vertices,
            'num_points': vertices.shape[0] if vertices.size > 0 else 0,
            'point_data': point_data
        }

    def _extract_faces_from_unstructured_grid(self, grid) -> Optional[np.ndarray]:
        """从非结构网格中提取三角形面片"""
        try:
            # 使用VTK的几何过滤器提取表面
            geometry_filter = vtk.vtkGeometryFilter()
            geometry_filter.SetInputData(grid)
            geometry_filter.Update()

            surface = geometry_filter.GetOutput()
            if not surface or surface.GetNumberOfPolys() == 0:
                # 如果没有表面多边形，尝试提取外部面
                dataset_surface_filter = vtk.vtkDataSetSurfaceFilter()
                dataset_surface_filter.SetInputData(grid)
                dataset_surface_filter.Update()
                surface = dataset_surface_filter.GetOutput()

            if not surface or surface.GetNumberOfPolys() == 0:
                print("  [WARNING] No surface polygons found in grid")
                return None

            # 提取三角形面片
            faces = []
            for i in range(surface.GetNumberOfCells()):
                cell = surface.GetCell(i)
                if cell.GetNumberOfPoints() == 3:  # 三角形
                    face_vertices = [cell.GetPointId(j) for j in range(3)]
                    faces.append(face_vertices)

            if len(faces) == 0:
                print("  [WARNING] No triangular faces found")
                return None

            return np.array(faces, dtype=np.int32)

        except Exception as e:
            print(f"  [WARNING] Failed to extract faces: {e}")
            return None

    def _vtk_array_to_numpy(self, vtk_array) -> np.ndarray:
        """将VTK数组转换为numpy数组"""
        if vtk_array.GetNumberOfComponents() == 1:
            # 标量场
            data = np.zeros(vtk_array.GetNumberOfTuples())
            for i in range(vtk_array.GetNumberOfTuples()):
                data[i] = vtk_array.GetComponent(i, 0)
        else:
            # 矢量场或张量场
            num_components = vtk_array.GetNumberOfComponents()
            num_tuples = vtk_array.GetNumberOfTuples()
            data = np.zeros((num_tuples, num_components))
            for i in range(num_tuples):
                for j in range(num_components):
                    data[i, j] = vtk_array.GetComponent(i, j)

        return data

    def get_available_fields(self, data: Dict[str, Any]) -> List[str]:
        """获取可用的物理场变量"""
        fields = []

        if 'blocks' in data:
            for block in data['blocks']:
                if 'point_data' in block:
                    fields.extend(block['point_data'].keys())
                if 'cell_data' in block:
                    fields.extend(block['cell_data'].keys())

        return list(set(fields))  # 去重


def test_vtk_reader():
    """测试VTK读取功能"""
    reader = VTKReader()

    # 读取示例VTM文件
    vtm_file = "Data/vessel.000170.vtm"
    try:
        data = reader.read_vtm(vtm_file)
        print(f"成功读取VTM文件，包含 {data['num_blocks']} 个数据块")

        # 显示可用场变量
        fields = reader.get_available_fields(data)
        print(f"可用物理场变量: {fields}")

        # 显示第一个数据块的信息
        if data['blocks']:
            first_block = data['blocks'][0]
            print(f"第一个数据块类型: {first_block['type']}")
            print(f"点数: {first_block['num_points']}")
            print(f"单元数: {first_block.get('num_cells', 'N/A')}")

    except Exception as e:
        print(f"读取失败: {e}")


if __name__ == "__main__":
    test_vtk_reader()