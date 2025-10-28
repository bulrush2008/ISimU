# ISimU (Intelligent Simulation for UNS)

基于深度学习的CFD代理模型开发平台，通过UNS求解器结果训练神经网络实现快速流场预测。

## 🎯 项目概述

ISimU是一个专业的CFD（计算流体动力学）代理模型开发平台，旨在将传统的UNS求解器仿真结果转换为深度学习可用的矩阵数据，并构建能够进行毫秒级流场预测的神经网络模型。

### 核心特性

- ✅ **VTK格式支持**：完整支持UNS求解器的VTM/VTU格式输出
- ✅ **SDF插值算法**：基于符号距离场的血管内外判断
- ✅ **多物理场支持**：压力、速度、CellID等多变量插值
- ✅ **智能域外处理**：血管外部点自动赋值为-1.0
- ✅ **矢量场插值**：支持3维速度场插值
- ✅ **多格式输出**：HDF5存储 + VTK可视化

## 📁 项目结构

```
ISimU/
├── src/                           # 核心源代码
│   ├── __init__.py
│   ├── data_reader.py             # VTK文件读取模块
│   ├── interpolator.py            # 网格插值模块（支持SDF）
│   ├── hdf5_storage.py            # HDF5存储模块
│   └── sdf_utils.py               # 符号距离场工具
├── examples/                      # 示例脚本
│   ├── complete_pipeline_en.py    # 完整处理流程（英文）
│   ├── complete_64x64x64_interpolation.py  # 64x64x64完整插值
│   ├── test_custom_interpolation.py        # 自定义插值方法测试
│   └── test_sdf_interpolation.py          # SDF插值测试
├── tests/                         # 测试文件
│   └── __init__.py
├── Data/                          # 原始CFD数据
│   └── vessel.000170.vtm          # 示例血管流数据
├── matrix_data/                   # 生成的矩阵数据
│   ├── vessel_170_64x64x64_complete.h5   # 64x64x64插值结果
│   └── vessel_170_64x64x64_complete.vts  # VTK可视化文件
├── pyproject.toml                 # 项目配置
├── CLAUDE.md                      # 详细需求文档
├── SDF_INTERPOLATION_UPDATE.md    # SDF插值更新总结
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
pip install vtk numpy scipy h5py pandas torch matplotlib trimesh
```

### 基本使用

#### 1. 简单插值示例

```python
from src.data_reader import VTKReader
from src.interpolator import GridInterpolator
from src.hdf5_storage import HDF5Storage

# 读取VTK文件
reader = VTKReader()
vtk_data = reader.read_vtm('Data/vessel.000170.vtm')

# 创建插值器（64x64x64网格，启用SDF）
interpolator = GridInterpolator(
    grid_size=(64, 64, 64),
    method='linear',
    out_of_domain_value=-1.0,
    use_sdf=True
)

# 执行插值
result = interpolator.interpolate(vtk_data, ['P', 'Velocity'])

# 保存结果
storage = HDF5Storage()
storage.save(result, 'output.h5')
```

#### 2. 完整64x64x64插值

```bash
# 运行完整的64x64x64网格插值
uv run python examples/complete_64x64x64_interpolation.py
```

#### 3. SDF插值测试

```bash
# 测试基于SDF的插值方法
uv run python examples/test_sdf_interpolation.py
```

## 🔧 核心功能模块

### 1. VTK数据读取 (`data_reader.py`)

支持多种VTK格式：
- **VTM多块数据集**：UNS求解器的主要输出格式
- **VTU非结构网格**：详细的网格和场数据
- **面片提取**：为SDF计算提供表面几何信息

```python
reader = VTKReader()
vtk_data = reader.read_vtm('file.vtm')
available_fields = reader.get_available_fields(vtk_data)
```

### 2. SDF插值器 (`interpolator.py` + `sdf_utils.py`)

#### 符号距离场算法
- **φ > 0**：血管内部，执行插值
- **φ < 0**：血管外部，直接赋值-1.0

#### 插值方法
- **标准方法**：linear、nearest、cubic
- **自定义方法**：
  - 最近邻直接赋值
  - 3点平均值

```python
interpolator = GridInterpolator(
    grid_size=(128, 128, 128),  # 默认网格尺寸
    out_of_domain_value=-1.0,   # 域外值
    use_sdf=True               # 启用SDF判断
)
```

### 3. HDF5存储 (`hdf5_storage.py`)

#### 数据结构
```
HDF5文件结构:
├── fields/                   # 物理场数据
│   ├── P                    # 压力场 (64,64,64)
│   ├── Velocity             # 速度场 (64,64,64,3)
│   └── CellID               # 单元ID (64,64,64)
├── grid/                    # 网格坐标
│   ├── x, y, z             # 笛卡尔坐标
└── metadata/                # 元数据信息
```

#### 格式转换
- **HDF5 → VTK**：用于ParaView可视化
- **元数据保存**：插值参数、原始文件信息等

## 📊 插值结果分析

### 64x64x64网格插值示例

| 字段 | 网格形状 | 有效插值点 | 域外点 | 有效值范围 |
|------|----------|------------|--------|------------|
| **压力P** | (64,64,64) | 46,515 (17.7%) | 215,629 (82.3%) | [-1.446e+02, 7.672e+02] |
| **速度Velocity** | (64,64,64,3) | 139,545 (17.7%) | 646,887 (82.3%) | [-6.843e-01, 7.314e-01] |
| **CellID** | (64,64,64) | 0 (0.0%) | 262,144 (100.0%) | [-1.0, -1.0] |

### 文件大小
- **HDF5数据**：1.03 MB
- **VTK可视化**：1.28 MB

## 🎨 可视化

生成的VTK文件可以在以下软件中打开：
- **ParaView**：推荐的专业可视化工具
- **VisIt**：科学可视化软件
- **其他VTK支持的工具**

```bash
# 使用ParaView打开
paraview matrix_data/vessel_170_64x64x64_complete.vts
```

## 🔬 技术特点

### SDF算法优势
1. **精确的几何判断**：基于符号距离场的血管内外识别
2. **物理意义明确**：域外值统一为-1.0，便于神经网络处理
3. **计算效率高**：使用KD树加速最近邻搜索

### 支持的数据类型
- **标量场**：压力、密度、温度等
- **矢量场**：速度、位移等（支持多维）
- **标识场**：CellID、NodeID等

### 插值精度
- **线性插值**：平衡精度和速度
- **最近邻**：保持原始数据特征
- **三次样条**：高精度插值（计算量大）

## 🚧 开发状态

### ✅ 已完成功能
- [x] VTK/VTM文件读取
- [x] 符号距离场(SDF)计算框架
- [x] 64×64×64网格插值
- [x] 压力和速度场插值
- [x] 矢量场支持
- [x] HDF5存储和VTK输出
- [x] 血管内外自动判断
- [x] 域外点处理

### 🚧 进行中功能
- [ ] 128×128×128默认网格插值
- [ ] SDF面片提取优化
- [ ] 批量数据处理
- [ ] 深度学习模型集成

### 📋 计划功能
- [ ] CNN代理模型训练
- [ ] 实时流场预测
- [ ] 模型评估和验证
- [ ] Web界面开发

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