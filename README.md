# ISimU (Intelligent Simulation for UNS)

基于深度学习的CFD代理模型开发平台，通过UNS求解器结果训练神经网络实现快速流场预测。

## 项目结构

```
ISimU/
├── src/                    # 源代码
│   ├── __init__.py
│   ├── data_reader.py      # VTK文件读取模块
│   ├── interpolator.py     # 网格插值模块
│   ├── hdf5_storage.py     # HDF5存储模块
│   └── visualizer.py       # 可视化模块
├── tests/                  # 测试文件
│   ├── __init__.py
│   └── test_interpolation.py
├── examples/               # 示例脚本
│   └── basic_example.py
├── requirements.txt        # 依赖包
├── README.md              # 项目说明
└── CLAUDE.md              # 项目需求文档
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```python
from src.data_reader import VTKReader
from src.interpolator import GridInterpolator
from src.hdf5_storage import HDF5Storage

# 读取VTK文件
reader = VTKReader()
data = reader.read_vtk('Data/vessel.000170.vtm')

# 插值到笛卡尔网格
interpolator = GridInterpolator(grid_size=(64, 64, 64))
cartesian_data = interpolator.interpolate(data)

# 保存为HDF5格式
storage = HDF5Storage()
storage.save(cartesian_data, 'output.h5')
```

## 核心功能

1. **矩阵数据库功能**：VTK→笛卡尔网格插值→HDF5存储
2. **深度学习代理模型**：CNN网络训练与预测