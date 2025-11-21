# ISimU (Intelligent Simulation for UNS)

基于深度学习的CFD代理模型开发平台，通过UNS求解器结果训练神经网络实现快速流场预测。

## 🎯 项目概述

ISimU是一个专业的CFD（计算流体动力学）代理模型开发平台，旨在将传统的UNS求解器仿真结果转换为深度学习可用的矩阵数据，并构建能够进行毫秒级流场预测的神经网络模型。

### 📊 **多数据源支持**

项目支持两种数据源：

#### **data_UNS/ (STL+VTU格式)**
- **几何文件**：STL格式血管几何 (`data_UNS/geo/portal_vein_A.stl`)
- **流场文件**：VTU格式CFD结果 (`data_UNS/vessel/170/Part.0.Zone.1.vtu`)
- **缩放处理**：自动应用0.001缩放比例

#### **data_VMR/ (VTP+VTU格式)** 🆕
- **配置文件**：`data_VMR/geo-flow.json` (7个算例配置)
- **几何文件**：VTP格式血管几何 (无需缩放)
- **流场文件**：VTU格式CFD结果
- **缩放处理**：无需缩放（比例=1.0）

**当前开发重点**：正在实施对VTP格式的完整支持 (详见CLAUDE.md中的实施计划)

### ✅ 已完成功能
- ✅ **VTK格式支持**：完整支持UNS求解器的VTM/VTU格式输出
- ✅ **STL几何集成**：基于真实血管几何的符号距离场计算
- ✅ **多数据源架构**：支持STL+VTU和VTP+VTU两种数据格式
- 🚧 **VTP格式支持**：正在开发中 (支持data_VMR/的7个算例)
- ✅ **精确SDF算法**：正确的符号距离场计算（内部>0，外部<0）
- ✅ **多物理场支持**：压力、速度、CellID、SDF等多变量插值
- ✅ **智能域外处理**：血管外部点自动赋值为0.0（压力和速度）
- ✅ **矢量场插值**：支持3维速度场插值
- ✅ **多格式输出**：HDF5存储 + VTK可视化
- ✅ **性能优化**：优化版本interpolator_optimized.py，消除SDF重复计算
- ✅ **多尺度网格**：16×16×16到64×64×64网格支持
- ✅ **内存优化**：批处理机制支持大网格计算

### 🚧 性能优化阶段（当前重点）
- ✅ **SDF重复计算优化**：消除重复SDF计算，实现1.67x性能提升
- ✅ **算法优化**：使用LinearNDInterpolator替代重复griddata调用
- ✅ **优化版本**：interpolator_optimized.py已实现并测试
- 🚧 **并行计算**：多进程SDF计算，多线程插值处理（开发中）
- 🚧 **批处理优化**：增大批处理大小，减少函数调用开销（开发中）

## 🔄 数据流程

ISimU采用**统一的数据流程**，确保数据一致性和可追溯性：

### 📊 **实际的数据流程**

```
原VTU数据 → 插值器 → HDF5矩阵数据 → VTK转换器 → VTS可视化文件
     ↓                                      ↓
  非结构网格                              结构化网格
  228,216点                              32,768点
```

### 🎯 **详细处理步骤**

1. **数据读取阶段**
   - 输入：`data_UNS/vessel/170/Part.0.Zone.1.vtu` (CFD仿真结果)
   - 读取：非结构网格，228,216个点，65,328个单元
   - 物理场：压力(P), 速度(Velocity), 密度(RHO), 节点ID(NodeID)

2. **SDF计算阶段**
   - 几何：`data_UNS/geo/portal_vein_A.stl` (血管几何，缩放0.001)
   - 计算：符号距离场，判断血管内外位置
   - 结果：φ>0内部，φ<0外部，精确几何边界

3. **智能插值阶段**
   - 网格：非结构网格 → 笛卡尔网格(32×32×32)
   - 策略：血管内部插值，血管外部赋零(P=0, Velocity=(0,0,0))
   - 结果：32,768个点的结构化流场数据

4. **数据存储阶段**
   - 格式：HDF5文件 (高效存储 + 完整元数据)
   - 内容：网格坐标 + 所有物理场 + SDF值
   - 大小：~1-2MB (32³网格) 到 ~8-10MB (64³网格)

5. **可视化转换阶段**
   - 输入：HDF5文件数据
   - 转换：VTK结构化网格格式
   - 输出：VTS文件 (ParaView兼容)
   - 验证：VTS文件时间戳比HDF5晚~0.03秒

### ✅ **数据一致性保证**

- **时间戳验证**：VTS比HDF5晚0.03秒，证明从HDF5生成
- **内容验证**：VTS包含与HDF5相同的物理场数据
- **网格一致性**：HDF5和VTS都有32,768个网格点
- **数据连续性**：插值结果在两个格式中完全一致

### 📁 **输出文件示例**

```bash
# HDF5矩阵数据（主要输出）
data_matrix/dense_32x32x32_zero_assignment_new_paths.h5
├── grid/           # 网格坐标 (32,32,32)
├── fields/         # 物理场数据
│   ├── P          # 压力场 (32,32,32)
│   ├── Velocity   # 速度场 (32,32,32,3)
│   ├── SDF        # 符号距离场 (32,32,32)
│   ├── RHO        # 密度场 (32,32,32)
│   └── NodeID     # 节点ID (32,32,32)
└── metadata/       # 处理信息

# VTK可视化文件（从HDF5生成）
data_matrix/dense_32x32x32_zero_assignment_new_paths.vts
```

## 📁 项目结构

```
ISimU/
├── src/                           # 核心源代码
│   ├── __init__.py
│   ├── data_reader.py             # VTK文件读取模块
│   ├── interpolator.py            # 原始网格插值模块（稳定版本）
│   ├── interpolator_optimized.py  # 优化网格插值模块（性能优化版本）
│   ├── hdf5_storage.py            # HDF5存储模块
│   ├── sdf_utils.py               # 符号距离场工具
│   └── stl_reader.py              # STL几何文件读取模块
├── examples/                      # 示例脚本
│   ├── quick_start.py                     # 快速开始示例
│   ├── complete_pipeline.py                # 完整流程示例
│   ├── complete_pipeline_en.py             # 完整流程示例（英文版）
│   ├── complete_64x64x64_interpolation.py  # 64x64x64完整插值
│   ├── test_dense_48x48x48.py              # 48x48x48密集网格测试（优化版本）
│   ├── test_sdf_16x16x16_corrected.py      # 16x16x16 SDF验证测试
│   ├── test_sdf_32x32x32_corrected.py      # 32x32x32 SDF验证测试
│   ├── test_zero_assignment.py             # 域外点赋零策略测试
│   ├── test_stl_sdf.py                      # STL-based SDF测试
│   ├── test_sdf_saving.py                   # SDF值保存测试
│   ├── test_custom_interpolation.py         # 自定义插值测试
│   ├── test_sdf_interpolation.py            # SDF插值测试
│   ├── test_sdf_small_grid.py               # 小网格SDF测试
│   ├── test_stl_path.py                     # STL路径测试
│   ├── test_sdf_32x32x32.py                 # 32x32x32 SDF测试
│   ├── test_vtk_reader.py                   # VTK读取器测试
│   └── INTERPOLATION_UPDATE_SUMMARY.md      # 插值更新总结
├── docs/                          # 文档目录
│   ├── *.md                        # 技术文档和更新总结
│   ├── performance_analysis*.txt   # 性能分析数据
│   └── project_development_summary.md
├── tests/                         # 测试模块
│   ├── __init__.py
│   └── test_interpolation.py       # 插值测试
├── Data/                          # 原始CFD数据
│   ├── vessel.000170.vtm          # 示例血管流数据
│   ├── vessel/                    # 血管流数据分解
│   │   └── 170/                   # 时间步170的数据
│   │       ├── Part.0.Zone.1.vtu # 主要流场数据
│   │       ├── Part.0.*.vtp      # 边界条件数据
│   │       └── *.vtm              # 各部分VTM文件
│   └── geo/                       # 几何数据
│       ├── portal_vein_A.stl     # 门静脉血管几何（缩放0.001）
│       ├── vessel.stp            # 血管几何（STEP格式）
│       └── scaling.txt           # 缩放配置
├── matrix_data/                   # 生成的矩阵数据
│   ├── dense_48x48x48_zero_assignment.h5     # 48x48x48密集网格结果
│   ├── dense_48x48x48_zero_assignment.vts    # VTK可视化文件
│   └── *.h5                       # 其他生成的HDF5数据
├── pyproject.toml                 # 项目配置
├── uv.lock                        # uv锁定文件
├── CLAUDE.md                      # 详细需求文档
├── README.md                      # 项目说明
```

## 🚀 快速开始

### 环境安装

本项目采用**uv**进行环境配置和依赖管理，这是推荐的方式：

```bash
# 安装依赖并创建虚拟环境
uv sync

# 激活虚拟环境（Windows）
.venv\Scripts\activate

# 激活虚拟环境（Linux/Mac）
source .venv/bin/activate

# 运行测试算例
uv run python examples/quick_start.py
```

或使用传统pip方式：

```bash
pip install vtk numpy scipy h5py pandas torch matplotlib trimesh rtree
```

**环境要求**：
- Python >= 3.8
- 推荐使用uv进行环境管理
- 支持Windows、Linux和macOS

### 基本使用

#### 1. 基于STL几何的SDF插值（推荐）

```python
from src.data_reader import VTKReader
from src.interpolator import GridInterpolator
from src.hdf5_storage import HDF5Storage

# 读取VTK文件
reader = VTKReader()
vtk_data = reader.read_vtm('Data/vessel.000170.vtm')

# 创建插值器（32x32x32网格，启用STL-based SDF）
interpolator = GridInterpolator(
    grid_size=(32, 32, 32),
    method='linear',
    out_of_domain_value=0.0,  # 域外点赋零
    use_sdf=True
)

# 执行插值（自动加载STL几何计算SDF）
result = interpolator.interpolate(vtk_data, ['P', 'Velocity'])

# 保存结果（包含SDF字段）
storage = HDF5Storage()
storage.save(result, 'output_with_sdf.h5')

# 转换为VTK可视化文件
storage.convert_to_vtk('output_with_sdf.h5', 'output_with_sdf.vts')
```

### 2. 优化版本插值器（性能提升1.67x）

```python
from src.interpolator_optimized import OptimizedGridInterpolator

# 使用优化版本插值器
interpolator = OptimizedGridInterpolator(
    grid_size=(48, 48, 48),
    method='linear',
    out_of_domain_value=0.0,
    use_sdf=True,
    batch_size=15000  # 增大批处理提升性能
)

# 执行优化插值（避免SDF重复计算）
result = interpolator.interpolate(vtk_data, ['P', 'Velocity', 'SDF'])
```

#### 3. STL几何文件处理

```python
from src.stl_reader import load_portal_vein_geometry

# 加载门静脉血管几何（自动缩放0.001）
stl_data = load_portal_vein_geometry()

print(f"几何信息：")
print(f"  - 顶点数：{stl_data['num_vertices']:,}")
print(f"  - 面片数：{stl_data['num_faces']:,}")
print(f"  - 缩放比例：{stl_data['scale_factor']}")
print(f"  - 是否水密：{stl_data['is_watertight']}")
```

#### 4. 测试和验证

```bash
# 快速开始和完整流程测试
uv run python examples/quick_start.py
uv run python examples/complete_pipeline.py

# SDF算法验证测试
uv run python examples/test_sdf_16x16x16_corrected.py
uv run python examples/test_sdf_32x32x32_corrected.py

# 域外点赋零策略测试
uv run python examples/test_zero_assignment.py

# 密集网格测试（推荐）
uv run python examples/test_dense_48x48x48.py      # 48×48×48网格
uv run python examples/test_dense_64x64x64.py      # 64×64×64网格

# 性能优化测试
uv run python examples/performance_analysis_48x48x48.py  # 性能对比分析
uv run python examples/compare_dense_performance.py      # 优化版本对比

# STL几何处理测试
uv run python examples/test_stl_sdf.py
uv run python examples/test_stl_path.py

# 其他功能测试
uv run python examples/test_custom_interpolation.py  # 自定义插值方法
uv run python examples/test_sdf_saving.py           # SDF值保存
uv run python examples/complete_64x64x64_interpolation.py  # 完整64网格流程
```

## 🔧 核心功能模块

### 1. STL几何读取 (`stl_reader.py`)

支持STL格式的血管几何文件：
- **自动缩放**：按照CLAUDE.md要求缩放0.001
- **路径解析**：智能检测项目根目录
- **几何验证**：检查网格质量和水密性

```python
from src.stl_reader import load_portal_vein_geometry

# 自动加载Data/geo/portal_vein_A.stl
stl_data = load_portal_vein_geometry()
vertices = stl_data['vertices']  # 已缩放的顶点坐标
faces = stl_data['faces']        # 面片索引
```

### 2. 符号距离场计算 (`sdf_utils.py`)

#### SDF定义
- **φ > 0**：血管内部，表示到血管壁的距离
- **φ < 0**：血管外部，表示到血管壁距离的负值
- **φ = 0**：血管壁表面

#### 技术特点
- **真实几何**：基于STL几何的精确SDF计算
- **批处理**：内存优化的分批计算机制
- **容错处理**：trimesh失败时自动切换到近似方法

```python
from src.sdf_utils import VascularSDF

# 创建SDF计算器
sdf = VascularSDF(vertices, faces)

# 计算查询点的SDF值
query_points = np.array([[0.15, 0.15, 0.15], [0.3, 0.3, 0.3]])
sdf_values = sdf.compute_sdf(query_points)  # 返回符号距离值
```

### 3. SDF插值器 (`interpolator.py`)

#### 插值逻辑
1. **计算所有网格点的SDF值**
2. **判断点位置**：SDF > 0（内部）或 SDF < 0（外部）
3. **分别处理**：
   - 内部点：执行正常插值
   - 外部点：直接赋值out_of_domain_value（默认0.0）

```python
interpolator = GridInterpolator(
    grid_size=(32, 32, 32),      # 网格尺寸
    method='linear',              # 插值方法
    out_of_domain_value=0.0,      # 域外值（压力P=0，速度=(0,0,0)）
    use_sdf=True                  # 启用SDF判断
)
```

### 4. HDF5存储 (`hdf5_storage.py`)

#### 数据结构
```
HDF5文件结构:
├── fields/                   # 物理场数据
│   ├── P                    # 压力场 (48,48,48)
│   ├── Velocity             # 速度场 (48,48,48,3)
│   ├── CellID               # 单元ID (48,48,48)
│   └── SDF                  # 符号距离场 (48,48,48)
├── grid/                    # 网格坐标
│   ├── x, y, z             # 笛卡尔坐标
└── metadata/                # 元数据信息
    ├── stl_file            # STL文件信息
    ├── scale_factor        # 缩放比例
    ├── sdf_used            # SDF使用状态
    └── grid_size           # 网格尺寸信息
```

#### 格式转换
- **HDF5 → VTK**：用于ParaView可视化
- **SDF可视化**：正确显示血管内外区域

## 📊 SDF插值结果分析

### 16x16x16网格测试结果

| 字段 | 网格形状 | 内部插值点 | 外部点 | 内部比例 | SDF范围 |
|------|----------|------------|--------|----------|----------|
| **压力P** | (16,16,16) | 103 | 3,993 | 2.51% | [-1.08e+02, 5.99e+02] |
| **速度Velocity** | (16,16,16,3) | 103 | 3,993 | 2.51% | [-6.34e-01, 7.22e-01] |
| **SDF** | (16,16,16) | 103 | 3,993 | 2.51% | [-6.80e-02, 5.36e-03] |

### 32x32x32网格测试结果

| 字段 | 网格形状 | 内部插值点 | 外部点 | 内部比例 | SDF范围 |
|------|----------|------------|--------|----------|----------|
| **压力P** | (32,32,32) | 825 | 31,743 | 2.53% | [-1.21e+02, 6.94e+02] |
| **速度Velocity** | (32,32,32,3) | 825 | 31,743 | 2.53% | [-6.77e-01, 7.27e-01] |
| **SDF** | (32,32,32) | 825 | 31,743 | 2.53% | [-6.80e-02, 6.80e-03] |

### 48x48x48密集网格测试结果

| 字段 | 网格形状 | 内部插值点 | 外部点 | 内部比例 | SDF范围 |
|------|----------|------------|--------|----------|----------|
| **压力P** | (48,48,48) | 3,079 | 107,513 | 2.78% | [-1.21e+02, 6.94e+02] |
| **速度Velocity** | (48,48,48,3) | 3,055 | 107,537 | 2.76% | [-6.77e-01, 7.27e-01] |
| **SDF** | (48,48,48) | 3,081 | 107,511 | 2.79% | [-6.80e-02, 6.80e-03] |

### 64x64x64超密集网格测试结果

| 字段 | 网格形状 | 内部插值点 | 外部点 | 内部比例 | SDF范围 |
|------|----------|------------|--------|----------|----------|
| **压力P** | (64,64,64) | 7,011 | 255,133 | 2.67% | [-1.94e+02, 7.52e+02] |
| **速度Velocity** | (64,64,64,3) | 6,953 | 255,191 | 2.65% | [-7.24e-01, 7.31e-01] |
| **SDF** | (64,64,64) | 7,012 | 255,132 | 2.67% | [-6.94e-02, 6.90e-03] |

### SDF值分布验证

#### 48x48x48密集网格
- **正值（内部）**：3,081个点 (2.79%)，范围[3.22e-06, 6.80e-03]
- **负值（外部）**：107,511个点 (97.21%)，范围[-6.80e-02, -1.64e-06]
- **零值（表面）**：0个点 (0.00%)

#### 64x64x64超密集网格
- **正值（内部）**：7,012个点 (2.67%)，范围[4.65e-09, 6.90e-03]
- **负值（外部）**：255,132个点 (97.33%)，范围[-6.94e-02, -6.44e-07]
- **零值（表面）**：0个点 (0.00%)

### 域外点赋零策略验证

#### 48x48x48密集网格
- **压力场**：外部点107,513个 (97.22%) 正确赋值为0
- **速度场**：外部点107,537个 (97.24%) 正确赋值为(0,0,0)
- **内部点**：保持真实CFD插值值

#### 64x64x64超密集网格
- **压力场**：外部点255,133个 (97.33%) 正确赋值为0
- **速度场**：外部点255,191个 (97.35%) 正确赋值为(0,0,0)
- **内部点**：保持真实CFD插值值

### 文件大小对比
- **HDF5数据**：
  - dense_48x48x48_zero_assignment.h5：1.00 MB
  - dense_64x64x64_zero_assignment.h5：2.30 MB
- **VTK可视化**：
  - dense_48x48x48_zero_assignment.vts：对应的可视化文件
  - dense_64x64x64_zero_assignment.vts：超高清可视化文件

## 🎨 SDF可视化

生成的VTK文件包含SDF字段，可以在ParaView中进行可视化：

```bash
# 使用ParaView打开SDF可视化文件
paraview matrix_data/dense_48x48x48_zero_assignment.vts
paraview matrix_data/dense_64x64x64_zero_assignment.vts  # 超高清版本
```

### 可视化指南
- **正值（红色）**：血管内部区域，显示真实插值值
- **负值（蓝色）**：血管外部区域，显示距离值
- **零值（绿色）**：血管壁表面
- **等值面**：可设置SDF=0的等值面显示血管几何

#### 高质量可视化（48x48x48密集网格）
推荐使用 `dense_48x48x48_zero_assignment.vts` 进行详细分析：
1. **加载文件**：直接打开VTS文件
2. **切片分析**：应用Slice过滤器查看不同截面的流场
3. **阈值过滤**：使用SDF > 0阈值显示仅血管内部
4. **流场可视化**：
   - 按压力P着色查看压力分布
   - 按速度Velocity magnitude查看流速分布
   - 按SDF着色查看血管几何边界

#### 超高清可视化（64x64x64超密集网格）
推荐使用 `dense_64x64x64_zero_assignment.vts` 进行超精细分析：
1. **超高分辨率**：2.37倍于48x48x48网格的点密度
2. **更精细的细节**：适合详细的流场结构分析
3. **高质量渲染**：生成更平滑的可视化效果
4. **精确边界检测**：更准确的血管壁表面重建

## 🔬 技术特点

### SDF算法优势
1. **真实几何**：基于STL几何的精确距离计算
2. **物理意义明确**：
   - 内部点：到血管壁的正距离
   - 外部点：到血管壁的负距离
3. **计算效率高**：批处理+KD树加速
4. **内存优化**：分批计算避免内存溢出

### 🔍 SDF计算原理详解

#### 核心计算流程
```
STL几何文件 → 顶点+面片数据 → Trimesh对象 → 符号距离场计算
```

#### 详细实现步骤

**第1步：STL几何数据提取**
```python
# 读取STL文件，获取几何数据
stl_data = load_portal_vein_geometry()
vertices = stl_data['vertices']     # 顶点坐标 (N, 3)
faces = stl_data['faces']         # 面片索引 (M, 3)
```

**第2步：Trimesh对象构建**
```python
class VascularSDF:
    def __init__(self, vertices, faces):
        self.vertices = vertices          # 存储顶点数据
        self.faces = faces                # 存储面片数据
        self.mesh = trimesh.Trimesh(vertices, faces)  # 构建trimesh网格对象
        self.kdtree = cKDTree(vertices)   # 构建KD树加速索引
        self.face_normals = self._compute_face_normals()  # 预计算面片法向量
```

**第3步：符号距离场计算**
```python
def _compute_sdf_trimesh(self, points: np.ndarray) -> np.ndarray:
    """使用trimesh计算精确SDF"""
    # 应用trimesh的最近表面查询和符号距离计算
    distances = self.mesh.nearest.signed_distance(points)
    return distances
    #
    # 内部逻辑：
    # 1. nearest：找到每个查询点的最近表面点
    # 2. signed_distance：计算欧几里得距离并判断内外方向
    # 3. 返回：正值（内部）或负值（外部）的距离值
```

#### SDF值定义
- **φ > 0**：血管内部，距离血管壁的正距离
- **φ < 0**：血管外部，距离血管壁的负距离
- **φ = 0**：血管壁表面

#### 计算方法对比

| 方法 | 精度 | 速度 | 使用场景 |
|------|------|------|----------|
| **Trimesh.signed_distance** | 高 | 中 | 主要方法（精确） |
| **KD树+法向量** | 中 | 快 | 备用方法（近似） |

### 🔍 Trimesh Signed Distance 算法深度解析

基于对trimesh 4.9.0源码的分析，其signed_distance算法采用了**混合策略**，结合了多种计算几何技术：

#### 📋 核心算法流程

```python
def signed_distance(mesh, points):
    # Step 1: 最近点查找
    closest, distance, triangle_id = closest_point(mesh, points)

    # Step 2: 符号计算（混合策略）
    # 2a. 法向量方法（投影在三角形内）
    # 2b. 光线投射方法（投影在三角形外）

    return signed_distance
```

#### 🎯 算法详细步骤

**Step 1: 最近点查找** (`closest_point`)
```python
# 空间加速结构
candidates = nearby_faces(mesh, points)  # 基于KD树的候选面筛选
triangles = mesh.triangles.view(np.ndarray)
# 在候选面中精确计算最近点
```

**关键技术特点**：
- ✅ **空间加速**：使用AABB/KD树快速筛选候选面片
- ✅ **精确计算**：最近点在**三角形面片上**，不是最近顶点
- ✅ **高效索引**：时间复杂度从O(n²)降到O(log n)

**Step 2: 符号计算（混合策略）**

**2a. 投影在三角形内 → 法向量方法**
```python
# 计算点到三角形平面的投影
projection = points - normals * dot(points - closest, normals)
# 重心坐标判断投影是否在三角形内
barycentric = points_to_barycentric(triangles, projection)
ontriangle = barycentric在三角形内
# 使用法向量确定符号
sign = sign(dot(normals, points - projection))
distance[ontriangle] *= -1.0 * sign
```

**2b. 投影在三角形外 → 光线投射方法**
```python
# 对剩余点使用光线投射判断内外
inside = mesh.ray.contains_points(points[remaining])
sign = (inside.astype(int) * 2) - 1.0  # 内部+1，外部-1
distance[remaining] *= sign
```

#### ⚡ 算法优势分析

| 特性 | Trimesh混合方法 | 简单KD树方法 |
|------|-----------------|--------------|
| **精度** | **极高**（精确几何） | 中等（近似） |
| **速度** | 中-高（混合优化） | 快速（简单查询） |
| **鲁棒性** | **极高**（双重容错） | 较低（易失败） |
| **复杂度** | O(log n) 平均 | O(log n) |
| **内存** | 中等 | 较低 |

#### 🚀 关键技术亮点

1. **智能混合策略**：
   - **90%+情况**：使用快速的法向量方法
   - **少数情况**：自动切换到可靠的光线投射

2. **多重容错机制**：
   - 法向量方法失败时自动降级
   - 确保算法的数值稳定性

3. **空间加速优化**：
   - 基于AABB/KD树的候选面筛选
   - 避免全局暴力搜索

4. **精确几何计算**：
   - 使用重心坐标进行精确的三角形包含测试
   - 最近点计算基于面片几何，非顶点近似

#### 📊 实际性能验证

根据ISimU项目的测试结果：
- **48³网格**：232 points/second
- **64³网格**：279 points/second
- **扩展效率**：120.6%（超线性扩展）

**性能特点**：
- ✅ **优秀的缓存局部性**（更大批处理更高效）
- ✅ **智能分治策略**（避免不必要计算）
- ✅ **数值优化实现**（使用einsum等高效操作）

#### 批处理机制
```python
def compute_sdf(self, points: np.ndarray, batch_size: int = 1000):
    """分批计算大量点的SDF，避免内存溢出"""
    if len(points) <= batch_size:
        return self._compute_sdf_trimesh(points)
    else:
        # 分批处理，每批1000个点
        sdf_values = np.zeros(len(points))
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            batch_sdf = self._compute_sdf_trimesh(batch_points)
            sdf_values[i:i+batch_size] = batch_sdf
        return sdf_values
```

#### 性能特点
- **空间复杂度**：O(N) - 与STL顶点数线性相关
- **时间复杂度**：O(log N) - KD树加速的最近邻查询
- **内存使用**：批处理机制，支持大规模网格计算
- **数值稳定性**：基于真实几何，避免体数据重建误差

### ⚡ 性能优化策略

#### ✅ **已实现的优化**
1. **消除重复计算**：
   - 修复SDF重复计算BUG，从2次减少到1次
   - 智能缓存机制，避免重复的符号距离场计算
   - 实际性能提升：**40%时间减少**

2. **算法层面**：
   - 使用LinearNDInterpolator预构建插值器，避免重复Delaunay三角剖分
   - 一次构建，多次查询，显著提升插值效率
   - 可配置批处理大小，减少函数调用开销

3. **内存优化**：
   - 分批计算机制，避免内存溢出
   - 及时释放临时变量，优化内存使用

#### 🚧 **进一步优化计划**
1. **并行计算**：
   - **多进程SDF**：并行计算符号距离场，充分利用多核CPU
   - **多线程插值**：对不同物理场并行插值处理

2. **批处理优化**：
   - 增大批处理大小（从1,000增加到15,000+）
   - 减少函数调用开销

3. **GPU加速（可选）**：
   - CuPy替换NumPy，GPU数组运算
   - RAPIDS cuML，GPU加速插值算法
   - 预期加速比：10-50x

### 📊 性能基准与优化效果

#### ✅ **已实现性能提升**
| 网格规模 | 优化前 | 优化后 | 性能提升 | 状态 |
|----------|--------|--------|----------|------|
| **48³ (11万点)** | 15分钟 | 9分钟 | **1.67x** | ✅ 已完成 |
| **64³ (26万点)** | 25-30分钟 | ~15-18分钟 | **1.67x** | ✅ 已完成 |

#### 🚧 **进一步优化目标**
| 网格规模 | 当前性能 | 目标性能 | 预期加速比 | 优化方法 |
|----------|----------|----------|------------|----------|
| 48³ (11万点) | 9分钟 | 2-3分钟 | 3-5x | 并行SDF+批处理 |
| 64³ (26万点) | 15-18分钟 | 5-8分钟 | 3-4x | 并行计算 |
| 128³ (200万点) | ~2小时 | ~30-60分钟 | 3-4x | GPU加速 |

#### 🎯 **优化策略**
- ✅ **算法优化**：使用LinearNDInterpolator替代重复griddata调用
- ✅ **消除重复计算**：修复SDF重复计算BUG，减少40%执行时间
- 🚧 **并行计算**：多进程SDF计算，多线程插值处理
- 🚧 **批处理优化**：增大批处理大小（1,000→15,000+）
- 📋 **GPU加速**：CuPy+RAPIDS（可选，预期10-50x加速）

### 几何处理能力
- **STL文件支持**：标准三角形网格格式
- **坐标缩放**：自动应用0.001缩放因子
- **水密性检查**：验证网格完整性
- **多尺度支持**：16x16x16到48x48x48密集网格
- **高分辨率计算**：110,592个点（48x48x48）的精确SDF计算

### 插值精度
- **线性插值**：平衡精度和速度
- **SDF指导**：基于几何位置的智能插值
- **边界处理**：精确的血管内外判断

## 🚧 开发状态

### ✅ 已完成功能
- [x] VTK/VTM文件读取
- [x] STL几何文件处理
- [x] 符号距离场(SDF)计算（基于真实几何）
- [x] SDF值存储和可视化
- [x] 多尺度网格支持（16x16x16, 32x32x32, 48x48x48）
- [x] 密集网格计算（110,592个点精确SDF）
- [x] 域外点智能赋零策略（P=0, Velocity=(0,0,0)）
- [x] 压力和速度场插值
- [x] 矢量场支持
- [x] HDF5存储和VTK输出
- [x] 血管内外精确判断
- [x] 内存优化的批处理机制
- [x] 坐标缩放（0.001倍）

### 🚧 性能优化阶段（当前重点）
- [x] 性能瓶颈分析和优化方案制定
- [x] **SDF重复计算优化**：消除重复SDF计算，实现1.67x性能提升
- [x] 算法优化：使用LinearNDInterpolator替代重复griddata调用
- [x] 优化版本开发：interpolator_optimized.py
- [ ] **并行计算**：多进程SDF计算，多线程插值处理
- [ ] **批处理优化**：增大批处理大小，减少函数调用开销
- [ ] **内存优化**：数据类型优化，流式处理
- [ ] **性能基准测试**：建立完整的性能评估体系

### 📋 计划功能
- [ ] GPU加速支持（CuPy + RAPIDS）
- [ ] 批量数据处理系统
- [ ] CNN代理模型训练
- [ ] 实时流场预测
- [ ] 模型评估和验证
- [ ] Web界面开发

## 🧪 测试和验证

### 运行测试
```bash
# 快速开始和完整流程测试
uv run python examples/quick_start.py
uv run python examples/complete_pipeline.py

# SDF算法验证测试
uv run python examples/test_sdf_16x16x16_corrected.py
uv run python examples/test_sdf_32x32x32_corrected.py

# 域外点赋零策略测试
uv run python examples/test_zero_assignment.py

# 48x48x48密集网格测试
uv run python examples/test_dense_48x48x48.py

# STL几何处理测试
uv run python examples/test_stl_sdf.py

# SDF值保存测试
uv run python examples/test_sdf_saving.py

# 自定义插值测试
uv run python examples/test_custom_interpolation.py

# 小网格SDF测试
uv run python examples/test_sdf_small_grid.py

# 性能优化测试（即将推出）
# uv run python examples/test_performance_optimization.py
# uv run python examples/test_parallel_interpolation.py
# uv run python examples/test_memory_optimization.py
```

### 验证要点
- ✅ **SDF符号正确性**：内部>0，外部<0
- ✅ **距离值合理性**：符合血管几何特征
- ✅ **插值一致性**：内外点处理正确
- ✅ **域外点赋零**：外部点P=0，Velocity=(0,0,0)
- ✅ **高分辨率计算**：48x48x48网格110,592个点精确计算
- ✅ **文件完整性**：HDF5和VTK文件正确生成

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 项目讨论区

---

**ISimU** - 让CFD仿真更快、更智能！