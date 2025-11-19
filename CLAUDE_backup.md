# ISimU 项目总控文档

## 项目概述
ISimU (Intelligent Simulation for UNS) 是一个基于深度学习的CFD代理模型开发平台，旨在通过UNS（基于非结构网格的CFD求解器）的仿真结果训练神经网络，构建能够快速预测流场的智能代理模型。

## 🚫 重要限制和禁止项

### Git操作限制
- **严禁任何自动git操作**：包括但不限于`git add`、`git commit`、`git push`、`git pull`、`git merge`等
- **git权限**：所有git操作必须由人工手动执行
- **版本控制**：代码更改完成后，由开发者自行决定是否提交到版本控制系统
- **分支管理**：禁止自动创建、切换或合并分支
- **标签操作**：禁止自动创建或删除git标签

## 🏗️ 项目架构

### 核心模块
ISimU采用模块化架构，包含以下核心模块：

```
ISimU/
├── src/
│   ├── data_processing/     # 数据处理模块
│   ├── neural_network/      # 神经网络模块
│   ├── inference/           # 推理预测模块
│   └── utils/               # 通用工具模块
├── examples/                # 示例代码
├── tests/                   # 测试代码
├── docs/                    # 模块文档
├── Data/                    # 原始数据
├── matrix_data/             # 处理后数据
├── models/                  # 训练模型
└── results/                 # 预测结果
```

### 模块功能概览

#### 1. 数据处理模块 (`data_processing/`)
**职责**：CFD原始数据到矩阵数据的转换
- **输入**：UNS求解器的VTK/VTU格式结果
- **输出**：标准化的HDF5矩阵数据
- **核心功能**：
  - VTK文件读取和解析
  - 非结构网格到笛卡尔网格插值
  - 基于STL几何的SDF计算
  - HDF5数据存储和管理
  - VTK可视化输出

**详细文档**：`docs/data_processing.md`

#### 2. 神经网络模块 (`neural_network/`)
**职责**：CFD代理模型的开发、训练和评估
- **输入**：HDF5矩阵数据
- **输出**：训练好的深度学习模型
- **核心功能**：
  - 3D CNN架构设计
  - 数据加载和预处理
  - 模型训练框架
  - 损失函数和评估指标
  - 模型保存和加载

**详细文档**：`docs/neural_network.md` (待创建)

#### 3. 推理预测模块 (`inference/`)
**职责**：使用训练好的模型进行流场快速预测
- **输入**：几何信息、边界条件
- **输出**：快速流场预测结果
- **核心功能**：
  - 模型预测器
  - 输入数据预处理
  - 输出结果后处理
  - 批量预测支持
  - 结果可视化

**详细文档**：`docs/inference.md` (待创建)

#### 4. 通用工具模块 (`utils/`)
**职责**：跨模块的通用功能
- **核心功能**：
  - 可视化工具
  - 配置管理
  - IO工具
  - 性能监控

**详细文档**：`docs/utils.md` (待创建)

## 🎯 项目目标

### 短期目标 (已完成)
- ✅ 完成VTK格式文件读取功能
- ✅ 开发非结构网格到均匀笛卡尔网格的插值算法
- ✅ 实现插值结果的HDF5格式存储
- ✅ 实现插值结果可视化
- ✅ 项目架构重构和模块化

### 中期目标 (进行中)
- 🔄 构建完整的矩阵数据库系统
- 🔄 批量处理多个VTK文件
- 🔄 建立3D CNN代理模型
- 🔄 实现模型训练和验证

### 长期目标
- 📋 实现毫秒级流场预测
- 📋 部署生产环境推理系统
- 📋 支持多种CFD求解器格式
- 📋 构建完整的MLOps工作流

## 🛠️ 技术栈

### 核心技术
- **编程语言**：Python 3.8+
- **深度学习框架**：PyTorch
- **数据处理**：NumPy, SciPy, VTK
- **数据存储**：HDF5, Pandas
- **几何处理**：trimesh, RTree
- **可视化**：Matplotlib, ParaView

### 依赖管理
- **包管理**：uv
- **环境隔离**：Python venv
- **版本控制**：Git

## 📁 关键文件和目录

### 配置文件
- `pyproject.toml` - 项目配置和依赖
- `uv.lock` - 锁定的依赖版本
- `CLAUDE.md` - 本文档，项目总控
- `README.md` - 项目说明文档

### 数据目录
- `Data/` - 原始CFD数据
  - `vessel.000170.vtm` - 示例流场数据
  - `geo/portal_vein_A.stl` - 血管几何文件
- `matrix_data/` - 处理后的矩阵数据
  - `dense_48x48x48_zero_assignment.h5` - 48³网格结果
  - `dense_64x64x64_zero_assignment.h5` - 64³网格结果

### 代码目录
- `src/` - 源代码
- `examples/` - 示例脚本
- `tests/` - 测试代码

## 📚 文档结构

### 模块文档
- `docs/data_processing.md` - 数据处理模块详细文档
- `docs/neural_network.md` - 神经网络模块文档 (待创建)
- `docs/inference.md` - 推理模块文档 (待创建)
- `docs/utils.md` - 通用工具文档 (待创建)

### 技术文档
- `docs/performance_analysis_v0.1.txt` - 性能分析数据
- `docs/performance_analysis_v0.2_64x64x64.txt` - 64³网格性能数据

## 🚀 快速导航

### 开发者指南
- **新手入门**：查看 `README.md` 快速开始
- **数据处理**：阅读 `docs/data_processing.md` 了解数据处理功能
- **神经网络**：查看 `src/neural_network/` 开始模型开发
- **测试验证**：运行 `tests/` 下的测试用例

### 数据流向
```
VTK/VTU (CFD结果)
    ↓ [data_processing]
HDF5 (矩阵数据)
    ↓ [neural_network]
PyTorch Model (训练模型)
    ↓ [inference]
快速流场预测
```

## 🔗 相关链接
- **项目仓库**：当前目录
- **示例代码**：`examples/`
- **API文档**：各模块的docstring
- **测试套件**：`tests/`

---

*最后更新：2025-11-14*
*版本：v2.0 - 模块化架构版本*