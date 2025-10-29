# ISimU (Intelligent Simulation for UNS)

基于深度学习的CFD代理模型开发平台，通过UNS求解器结果训练神经网络实现快速流场预测。

## 🎯 项目概述

ISimU是一个专业的CFD（计算流体动力学）代理模型开发平台，旨在将传统的UNS求解器仿真结果转换为深度学习可用的矩阵数据，并构建能够进行毫秒级流场预测的神经网络模型。

### 核心特性

- ✅ **VTK格式支持**：完整支持UNS求解器的VTM/VTU格式输出
- ✅ **STL几何集成**：基于真实血管几何的符号距离场计算
- ✅ **精确SDF算法**：正确的符号距离场计算（内部>0，外部<0）
- ✅ **多物理场支持**：压力、速度、CellID等多变量插值
- ✅ **智能域外处理**：血管外部点自动赋值为0.0（压力和速度）
- ✅ **矢量场插值**：支持3维速度场插值
- ✅ **多格式输出**：HDF5存储 + VTK可视化
- ✅ **内存优化**：批处理机制支持大网格计算

## 📁 项目结构

```
ISimU/
├── src/                           # 核心源代码
│   ├── __init__.py
│   ├── data_reader.py             # VTK文件读取模块
│   ├── interpolator.py            # 网格插值模块（支持SDF）
│   ├── hdf5_storage.py            # HDF5存储模块
│   ├── sdf_utils.py               # 符号距离场工具
│   └── stl_reader.py              # STL几何文件读取模块
├── examples/                      # 示例脚本
│   ├── complete_64x64x64_interpolation.py        # 64x64x64完整插值
│   ├── test_sdf_16x16x16_corrected.py            # 16x16x16 SDF验证测试
│   ├── test_sdf_32x32x32_corrected.py            # 32x32x32 SDF验证测试
│   ├── test_dense_48x48x48.py                   # 48x48x48密集网格测试
│   ├── test_zero_assignment.py                  # 域外点赋零策略测试
│   ├── test_stl_sdf.py                           # STL-based SDF测试
│   └── test_sdf_saving.py                        # SDF值保存测试
├── Data/                          # 原始CFD数据
│   ├── vessel.000170.vtm          # 示例血管流数据
│   └── geo/                         # 几何数据
│       └── portal_vein_A.stl       # 门静脉血管几何（缩放0.001）
├── matrix_data/                   # 生成的矩阵数据
│   ├── vessel_170_sdf_16x16x16_corrected.h5     # 16x16x16 SDF插值结果
│   ├── vessel_170_sdf_32x32x32_corrected.h5     # 32x32x32 SDF插值结果
│   ├── test_zero_assignment.h5                   # 域外点赋零测试结果
│   ├── dense_48x48x48_zero_assignment.h5        # 48x48x48密集网格结果
│   └── *.vts                                    # VTK可视化文件
├── pyproject.toml                 # 项目配置
├── CLAUDE.md                      # 详细需求文档
└── README.md                      # 项目说明
```

## 🚀 快速开始

### 环境安装

使用uv包管理器：

```bash
# 安装依赖
uv sync

# 激活虚拟环境（Windows）
.venv\Scripts\activate

# 激活虚拟环境（Linux/Mac）
source .venv/bin/activate
```

或使用pip：

```bash
pip install vtk numpy scipy h5py pandas torch matplotlib trimesh rtree
```

### 基本使用

#### 1. 基于STL几何的SDF插值

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

#### 2. STL几何文件处理

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

#### 3. SDF插值测试

```bash
# 测试16x16x16网格的SDF算法
uv run python examples/test_sdf_16x16x16_corrected.py

# 测试32x32x32网格的SDF算法
uv run python examples/test_sdf_32x32x32_corrected.py

# 测试域外点赋零策略
uv run python examples/test_zero_assignment.py

# 测试48x48x48密集网格
uv run python examples/test_dense_48x48x48.py

# 测试完整的STL-based SDF流程
uv run python examples/test_stl_sdf.py
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
│   ├── P                    # 压力场 (32,32,32)
│   ├── Velocity             # 速度场 (32,32,32,3)
│   ├── CellID               # 单元ID (32,32,32)
│   └── SDF                  # 符号距离场 (32,32,32) ← 新增
├── grid/                    # 网格坐标
│   ├── x, y, z             # 笛卡尔坐标
└── metadata/                # 元数据信息
    ├── stl_file            # STL文件信息
    ├── scale_factor        # 缩放比例
    └── sdf_used            # SDF使用状态
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

### SDF值分布验证（48x48x48密集网格）
- **正值（内部）**：3,081个点 (2.79%)，范围[3.22e-06, 6.80e-03]
- **负值（外部）**：107,511个点 (97.21%)，范围[-6.80e-02, -1.64e-06]
- **零值（表面）**：0个点 (0.00%)

### 域外点赋零策略验证
- **压力场**：外部点107,513个 (97.22%) 正确赋值为0
- **速度场**：外部点107,537个 (97.24%) 正确赋值为(0,0,0)
- **内部点**：保持真实CFD插值值

### 文件大小对比
- **HDF5数据**：
  - 16x16x16：0.07 MB
  - 32x32x32：0.28 MB
  - 48x48x48：1.00 MB
- **VTK可视化**：对应的.vts文件

## 🎨 SDF可视化

生成的VTK文件包含SDF字段，可以在ParaView中进行可视化：

```bash
# 使用ParaView打开SDF可视化文件
paraview matrix_data/vessel_170_sdf_16x16x16_corrected.vts
paraview matrix_data/dense_48x48x48_zero_assignment.vts
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

## 🔬 技术特点

### SDF算法优势
1. **真实几何**：基于STL几何的精确距离计算
2. **物理意义明确**：
   - 内部点：到血管壁的正距离
   - 外部点：到血管壁的负距离
3. **计算效率高**：批处理+KD树加速
4. **内存优化**：分批计算避免内存溢出

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

### 🚧 进行中功能
- [ ] 64×64×64网格优化
- [ ] 批量数据处理
- [ ] SDF计算性能优化
- [ ] 更多几何格式支持

### 📋 计划功能
- [ ] CNN代理模型训练
- [ ] 实时流场预测
- [ ] 模型评估和验证
- [ ] Web界面开发

## 🧪 测试和验证

### 运行测试
```bash
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