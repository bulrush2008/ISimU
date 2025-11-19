# 数据处理模块文档

## 模块概述
数据处理模块(`data_processing`)是ISimU项目的核心模块，负责将UNS求解器的VTK/VTU格式仿真结果转换为标准化的HDF5矩阵数据，为后续的神经网络训练提供高质量的输入数据。

## 模块功能

### 核心功能
数据处理模块包含以下核心功能：

#### 1. 矩阵数据库功能
- **数据提取**：从UNS求解器的VTK格式仿真结果中提取流场数据。目前的数据格式为vtm, vtu
- **网格插值**：将非结构网格数据插值到均匀笛卡尔网格上
- **数据存储**：将插值后的数据存储为HDF5格式的矩阵数据
- **数据管理**：建立索引和组织大量CFD仿真数据集
- **数据显示**：将矩阵数据转化为VTK数据，能在paraview上显示

### 详细技术规格

#### 网格插值设置方法
- **默认方法**：使用临近最近的网格直接赋值
- **可选方法**：用户可以选择使用临界的3个网格的平均值进行插值

#### 几何规格
- **几何数据**：`Data/geo/portal_vein_A.stl`，缩放比例0.001
- **流场数据**：位于`Data/vtm`文件，及关联的vtu文件，非结构网格
- **笛卡尔网格**：参照vtm流场数据的尺寸，基于按比例缩放后的stl几何文件
- **网格规模**：用于插值的笛卡尔网格默认为$64\times 64\times 64$

#### 区域区分策略
对于任意笛卡尔网格点：
- **血管内部**（$\phi>0$）：使用原数据插值
- **血管外部**（$\phi<0$）：直接赋值"0"（压力P=0，速度=(0,0,0)）

#### 插值步骤
1. **SDF计算**：计算符号距离SDF值$\phi$
   - SDF是Signed Distance Function的缩写
   - $|\phi|$绝对值表示距离血管壁的距离
   - $\phi>0$表示位于血管内部，需要使用血管流数据进行插值
   - $\phi<0$表示位于血管外，不进行插值，直接赋值0

2. **按符号插值**：
   - 血管内部$\phi>0$：使用流场数据插值
   - 血管外部$\phi<0$：直接赋值0

## 技术实现

### 编程语言
- **主语言**：Python 3.8+

### 核心依赖库
- **VTK**：读取和处理UNS的VTK格式结果
- **NumPy/SciPy**：数值计算和插值算法
- **h5py**：HDF5文件读写
- **pandas**：数据管理和索引
- **trimesh**：STL几何处理和SDF计算
- **rtree**：空间索引加速

### 数据格式
- **输入**：VTK格式（UNS求解器结果）
- **输出**：HDF5格式（矩阵数据库）
- **可视化**：VTK格式（ParaView兼容）

### 插值算法
- **线性插值**：LinearNDInterpolator
- **最近邻插值**：NearestNDInterpolator
- **径向基函数插值**：可通过第三方Python库提供
- **自定义方法**：
  - 最近点直接赋值
  - 临界3点平均插值

## 模块组件

### 文件结构
```
src/data_processing/
├── __init__.py                 # 模块入口
├── vtk_reader.py              # VTK文件读取模块
├── interpolator.py             # 原始网格插值模块（稳定版本）
├── interpolator_optimized.py   # 优化网格插值模块（性能优化版本）
├── sdf_utils.py                # 符号距离场工具
├── stl_reader.py               # STL几何文件读取模块
└── hdf5_storage.py             # HDF5存储模块
```

### 主要类和函数

#### VTKReader (`vtk_reader.py`)
- **功能**：读取UNS求解器的VTM/VTU格式文件
- **主要方法**：
  - `read_vtm(file_path)`: 读取VTM多块数据集
  - `read_vtu(file_path)`: 读取VTU非结构网格数据
  - `get_available_fields(data)`: 获取可用的场变量
  - `extract_point_data(block, field_name)`: 提取点数据

#### GridInterpolator (`interpolator.py`)
- **功能**：非结构网格到笛卡尔网格的插值
- **主要方法**：
  - `interpolate(vtk_data, field_names)`: 执行插值
  - `setup_cartesian_grid(bounds, grid_size)`: 设置笛卡尔网格
  - `get_interpolation_statistics(result)`: 获取插值统计信息

#### OptimizedGridInterpolator (`interpolator_optimized.py`)
- **功能**：性能优化的插值器，避免重复计算
- **优化特性**：
  - 使用LinearNDInterpolator预构建插值器
  - 缓存SDF计算结果
  - 批处理机制

#### VascularSDF (`sdf_utils.py`)
- **功能**：基于STL几何的符号距离场计算
- **主要方法**：
  - `compute_sdf(points)`: 计算点的SDF值
  - `batch_compute_sdf(points, batch_size)`: 批量计算SDF
  - `create_sdf_from_vtk_data(vtk_data)`: 从VTK数据创建SDF

#### HDF5Storage (`hdf5_storage.py`)
- **功能**：HDF5格式的数据存储和管理
- **主要方法**：
  - `save(data, file_path, metadata)`: 保存数据到HDF5
  - `load(file_path)`: 从HDF5加载数据
  - `get_file_info(file_path)`: 获取文件信息
  - `convert_to_vtk(h5_path, vtk_path)`: 转换为VTK格式

## 数据说明

### VTM文件格式
- **来源**：UNS求解器的多块数据集输出格式
- **内容**：
  - 流场变量：速度、压力、温度等
  - 非结构网格拓扑和几何信息
- **示例**：`Data/vessel.000170.vtm`（时间步170的结果）

### 场变量类型
- **标量场**：压力(P)、密度(RHO)、CellID、NodeID
- **矢量场**：速度(Velocity)
- **数据类型**：float64，支持多分量矢量

### HDF5输出格式
```python
# HDF5文件结构
{
    'fields': {
        'P': (grid_size),           # 压力场
        'Velocity': (grid_size, 3), # 速度场
        'SDF': (grid_size),         # 符号距离场
    },
    'grid': {
        'x': (grid_size),           # X坐标网格
        'y': (grid_size),           # Y坐标网格
        'z': (grid_size),           # Z坐标网格
    },
    'metadata': {                   # 元数据
        'grid_size': grid_size,
        'bounds': (xmin, xmax, ymin, ymax, zmin, zmax),
        'source_file': 'vessel.000170.vtm',
        'interpolation_method': 'linear',
        'zero_assignment_strategy': 'outside_vessel'
    }
}
```

## 性能要求

### 计算规模
- **单个算例**：
  - 小规模：16×16×16网格（4,096点）
  - 中等规模：48×48×48网格（110,592点）
  - 大规模：64×64×64网格（262,144点）
  - 超大规模：128×128×128网格（~200万点）
- **数据库规模**：数百个CFD算例的批量处理

### 性能目标
- **处理时间**：单个算例处理时间从数十秒优化到数秒
- **内存效率**：支持大规模数据集的内存管理
- **并行支持**：可选的GPU计算加速

### 硬件资源要求
- **CPU**：多核处理器，支持并行计算
- **内存**：
  - 64×64×64网格：至少2GB RAM
  - 128×128×128网格：至少8GB RAM
- **GPU**：可选，用于大规模计算加速

## 性能优化方向

### 算法优化
- **已完成**：使用LinearNDInterpolator替代重复的griddata调用
- **进行中**：优化插值算法和数据结构

### 并行计算
- **多进程SDF计算**：并行处理SDF值计算
- **多线程插值处理**：并行处理插值算法
- **批处理优化**：调整批处理大小以提升效率

### GPU加速
- **CuPy集成**：使用CuPy进行GPU数值计算
- **RAPIDS支持**：集成RAPIDS生态系统进行加速

### 内存优化
- **流式处理**：支持超大规模数据集的流式处理
- **数据类型优化**：使用合适的数据类型减少内存占用
- **批处理大小调优**：根据硬件配置优化批处理大小

## 当前状态

### 已实现功能 ✅
- VTK/VTM文件读取功能
- STL几何文件处理（支持0.001缩放）
- 基于STL的SDF计算（内部>0，外部<0）
- 域外点赋零策略（P=0，Velocity=(0,0,0)）
- 压力P和速度V的插值计算
- 多尺度网格支持（16×16×16到64×64×64）
- HDF5存储和VTK可视化
- 性能优化版本（LinearNDInterpolator）

### 生成的数据
- `matrix_data/dense_48x48x48_zero_assignment.h5` (1.0 MB)
- `matrix_data/dense_64x64x64_zero_assignment.h5` (2.4 MB)
- 对应的VTK可视化文件

### 性能基准
- **64×64×64网格**：262,144点，处理完成
- **SDF分布**：2.67%点在血管内部，97.33%在外部
- **插值质量**：
  - 压力范围：[-194.21, 751.78] Pa
  - 速度范围：[0, 0.96] m/s
- **处理策略**：批处理优化，每次处理1,000点

### 开发环境
- **操作系统**：Windows环境
- **环境管理**：使用uv配置环境和执行测试算例
- **版本控制**：Git，严禁自动操作

## 使用示例

### 基本使用流程
```python
from data_processing import VTKReader, GridInterpolator, HDF5Storage

# 1. 读取VTK数据
reader = VTKReader()
vtk_data = reader.read_vtm('Data/vessel.000170.vtm')

# 2. 设置插值器
interpolator = GridInterpolator(grid_size=(64, 64, 64))

# 3. 执行插值
available_fields = reader.get_available_fields(vtk_data)
result = interpolator.interpolate(vtk_data, ['P', 'Velocity'])

# 4. 保存结果
storage = HDF5Storage()
storage.save(result, 'matrix_data/output.h5')
```

### 性能优化版本
```python
from data_processing import VTKReader, OptimizedGridInterpolator, HDF5Storage

# 使用优化插值器
interpolator = OptimizedGridInterpolator(grid_size=(64, 64, 64))
# 其余步骤相同，但性能更好
```

## 示例脚本

### 快速开始
- `examples/data_processing/quick_start.py` - 基本功能演示
- `examples/data_processing/complete_pipeline.py` - 完整流程示例

### 测试脚本
- `examples/data_processing/test_dense_48x48x48.py` - 48³网格测试
- `examples/data_processing/test_dense_64x64x64.py` - 64³网格测试
- `examples/data_processing/performance_analysis_48x48x48.py` - 性能分析

### 功能测试
- `examples/data_processing/test_zero_assignment.py` - 域外赋零策略测试
- `examples/data_processing/test_sdf_*.py` - SDF相关测试
- `examples/data_processing/test_stl_*.py` - STL几何测试

## 测试套件

### 单元测试
- `tests/data_processing/test_interpolation.py` - 插值功能完整测试

### 测试覆盖
- VTK文件读取测试
- 网格插值功能测试
- HDF5存储测试
- 完整流程测试
- 插值质量测试

## 常见问题

### Q: SDF计算很慢怎么办？
A: 使用`OptimizedGridInterpolator`，它采用了批处理和缓存机制来加速SDF计算。

### Q: 内存不足如何处理？
A: 1) 减小网格尺寸；2) 使用更小的批处理大小；3) 启用内存优化模式。

### Q: 如何提高插值精度？
A: 1) 增加网格密度；2) 使用更高阶的插值方法；3) 优化STL几何质量。

### Q: 支持哪些VTK格式？
A: 目前主要支持UNS求解器的VTM多块数据集和VTU非结构网格。

## 开发指南

### 添加新的插值方法
1. 在`interpolator.py`中添加新的插值函数
2. 更新`interpolator_optimized.py`中的优化版本
3. 添加相应的测试用例
4. 更新文档说明

### 性能优化建议
1. 首先分析性能瓶颈（使用`performance_analysis_*.py`）
2. 考虑使用`OptimizedGridInterpolator`
3. 针对特定场景调优批处理大小
4. 必要时考虑GPU加速

### 调试技巧
1. 使用小网格尺寸（16×16×16）进行快速调试
2. 启用详细日志输出
3. 使用ParaView可视化中间结果
4. 运行单元测试验证功能正确性

---

*模块版本：v1.0*
*最后更新：2025-11-14*
*维护者：ISimU开发团队*