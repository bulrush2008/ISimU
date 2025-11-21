# 几何文件读取器架构设计

## 📋 设计目标

为ISimU项目设计统一的几何文件读取架构，支持多种几何格式，同时保持最小的影响和向后兼容性。

## 🏗️ 架构概览

### 核心组件

```
插值器调用路径：
interpolator.py → sdf_utils.py → geometry_reader.py
     ↓              ↓               ↓
  插值逻辑      SDF计算        几何数据读取
     ↓              ↓               ↓
  HDF5输出    血管内外判断    STL/VTP文件处理
```

### 模块职责

1. **geometry_reader.py** - 新增模块
   - 统一几何文件读取接口
   - 支持STL和VTP格式
   - 自动缩放处理
   - 数据格式标准化

2. **sdf_utils_enhanced.py** - 增强模块
   - 增强的SDF创建函数
   - 支持多种几何源
   - 向后兼容现有接口
   - VMR配置支持

3. **现有模块** - 保持不变
   - `interpolator.py` / `interpolator_optimized.py` - 插值逻辑
   - `data_reader.py` - VTU流场读取
   - `hdf5_storage.py` - 数据存储

## 🔧 技术设计

### 统一几何读取器 (geometry_reader.py)

```python
class GeometryReader:
    """统一几何文件读取器"""

    def read_geometry(file_path, scale_factor=None):
        """统一接口，自动检测格式"""
        if file_path.endswith('.stl'):
            return self._read_stl(file_path, scale_factor or 0.001)
        elif file_path.endswith('.vtp'):
            return self._read_vtp(file_path, scale_factor or 1.0)
```

**关键特性：**
- ✅ **自动格式检测**：基于文件扩展名
- ✅ **智能缩放**：STL默认0.001，VTP默认1.0
- ✅ **统一数据格式**：不同格式输出相同结构
- ✅ **错误处理**：完善的异常处理机制

### 增强SDF计算器 (sdf_utils_enhanced.py)

```python
class EnhancedSDFCalculator:
    """增强的SDF计算器"""

    def create_sdf_from_file(geometry_path, scale_factor=None):
        """从几何文件创建SDF"""

    def create_sdf_from_vmr_config(case_name, config_path):
        """从VMR配置创建SDF"""

    def create_sdf_enhanced(geometry_source, source_type='auto'):
        """通用SDF创建函数"""
```

**支持的输入源：**
- ✅ **STL文件路径**：`"path/to/file.stl"`
- ✅ **VTP文件路径**：`"path/to/file.vtp"`
- ✅ **VMR算例名称**：`"0007_H_AO_H"`
- ✅ **直接几何数据**：`{'vertices': ..., 'faces': ...}`

## 📊 数据流对比

### 现有架构 (仅STL)
```
STL文件 → stl_reader.py → VascularSDF → 插值器
```

### 新架构 (STL + VTP)
```
几何文件 → geometry_reader.py → VascularSDF → 插值器
   ↑                ↑
STL/VTP         统一接口
```

## 🔍 向后兼容性

### 现有代码无需修改

```python
# 现有的SDF创建方式仍然有效
from sdf_utils import create_sdf_from_vtk_data
sdf = create_sdf_from_vtk_data(vtk_data)
```

### 新的灵活方式

```python
# 新的增强方式
from sdf_utils_enhanced import create_sdf_enhanced

# STL文件
sdf = create_sdf_enhanced("geometry.stl")

# VTP文件
sdf = create_sdf_enhanced("geometry.vtp")

# VMR配置
sdf = create_sdf_enhanced("0007_H_AO_H", source_type='vmr_config')
```

## 🎯 集成策略

### 第一阶段：新增模块（当前）
- ✅ 创建 `geometry_reader.py`
- ✅ 创建 `sdf_utils_enhanced.py`
- ✅ 保持现有模块不变

### 第二阶段：逐步迁移（后续）
- 🔄 更新插值器使用新接口（可选）
- 🔄 废弃旧的STL专用代码（可选）
- 🔄 完全集成VMR支持

## 📋 文件清单

### 新增文件
- `src/geometry_reader.py` - 统一几何读取器
- `src/sdf_utils_enhanced.py` - 增强SDF计算器
- `docs/geometry_reader_architecture.md` - 架构文档

### 修改文件（最小影响）
- `src/sdf_utils.py` - 添加增强接口引用（可选）
- `src/interpolator_*.py` - 可选使用新接口

### 保持不变
- `src/data_reader.py` - VTU流场读取
- `src/hdf5_storage.py` - 数据存储
- `src/stl_reader.py` - STL专用读取器

## 🚀 使用示例

### 基本使用
```python
from geometry_reader import GeometryReader

reader = GeometryReader()

# 自动检测格式
geometry = reader.read_geometry("path/to/geometry.stl")  # STL
geometry = reader.read_geometry("path/to/geometry.vtp")  # VTP
```

### SDF创建
```python
from sdf_utils_enhanced import create_sdf_enhanced

# 多种输入方式
sdf1 = create_sdf_enhanced("geometry.stl")  # STL文件
sdf2 = create_sdf_enhanced("geometry.vtp")  # VTP文件
sdf3 = create_sdf_enhanced("0007_H_AO_H")   # VMR配置
```

## ✅ 验证清单

- [x] STL格式读取（测试通过）
- [x] VTP格式读取（待测试）
- [x] 缩放处理（自动检测）
- [x] 统一数据格式
- [x] 向后兼容性
- [x] 错误处理
- [ ] VMR配置支持（待测试）
- [ ] 集成测试（待测试）

## 🔮 扩展性

该架构设计支持未来扩展：

1. **新几何格式**：只需在 `GeometryReader` 中添加新的读取方法
2. **新配置方式**：在 `EnhancedSDFCalculator` 中添加新的创建方法
3. **新数据源**：通过统一的 `create_sdf_enhanced` 接口支持

## 📝 总结

该架构设计实现了：
- ✅ **统一接口**：支持多种几何格式
- ✅ **最小影响**：现有代码无需修改
- ✅ **向后兼容**：保持现有功能完整
- ✅ **扩展性强**：支持未来格式扩展
- ✅ **维护性好**：清晰的模块职责分离