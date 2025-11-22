# ISimU 可配置网格插值应用

基于`test_32x32x32_quick_verification.py`开发的用户可配置插值工具，支持自定义网格大小和多种参数。

主程序文件：`app_interpolation_uns.py`

## 功能特性

- ✅ **可配置网格大小**: 支持任意N×M×K网格规模
- ✅ **多种插值方法**: 线性插值、最近邻插值
- ✅ **SDF智能处理**: 基于STL几何的符号距离场计算
- ✅ **批处理优化**: 可调节批处理大小优化内存使用
- ✅ **双格式输出**: HDF5数据文件 + VTK可视化文件
- ✅ **性能统计**: 详细的处理速度和内存效率报告
- ✅ **安全检查**: 大规模网格前的确认提示

## 安装要求

确保已通过uv配置好项目环境：
```bash
uv sync
```

## 使用方法

### 基本用法

```bash
# 使用默认32×32×32网格
cd app
uv run python app_interpolation_uns.py

# 自定义网格大小
uv run python app_interpolation_uns.py --grid-size 64 64 64

# 使用不同的插值方法
uv run python app_interpolation_uns.py --grid-size 48 48 48 --method nearest

# 指定输出文件前缀
uv run python app_interpolation_uns.py --grid-size 32 32 32 --output-prefix my_simulation
```

### 高级选项

```bash
# 调整批处理大小（适用于不同内存配置）
uv run python app_interpolation_uns.py --grid-size 64 64 64 --batch-size 15000

# 禁用SDF计算（对所有点进行插值）
uv run python app_interpolation_uns.py --grid-size 32 32 32 --no-sdf

# 查看完整帮助
uv run python app_interpolation_uns.py --help
```

## 参数说明

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--grid-size` | `-g` | 3个整数 | 32 32 32 | 网格尺寸 (NX NY NZ) |
| `--method` | `-m` | 字符串 | linear | 插值方法 (linear/nearest) |
| `--batch-size` | `-b` | 整数 | 10000 | SDF计算批处理大小 |
| `--output-prefix` | `-o` | 字符串 | dense_interpolation | 输出文件名前缀 |
| `--no-sdf` | - | 标志 | False | 禁用SDF计算 |

## 输出文件

生成的文件保存在`../data_matrix/`目录下：

```
data_matrix/
├── {output-prefix}_{grid_size}.h5    # HDF5数据文件
└── {output-prefix}_{grid_size}.vts   # VTK可视化文件
```

例如：
```
data_matrix/
├── dense_interpolation_64x64x64.h5
└── dense_interpolation_64x64x64.vts
```

## 性能参考

| 网格规模 | 点数 | 预估时间 | 内存使用 | 适用场景 |
|----------|------|----------|----------|----------|
| 16×16×16 | 4,096 | ~3分钟 | ~70MB | 快速测试 |
| 32×32×32 | 32,768 | ~5分钟 | ~350KB | 标准处理 |
| 48×48×48 | 110,592 | ~15分钟 | ~1MB | 高质量结果 |
| 64×64×64 | 262,144 | ~30分钟 | ~2.3MB | 超精细分析 |

*注：实际性能取决于硬件配置和数据复杂度*

## 使用示例

### 示例1：快速测试
```bash
# 16×16×16网格，快速验证功能
uv run python app_interpolation_uns.py -g 16 16 16 -o quick_test
```

### 示例2：标准处理
```bash
# 48×48×48网格，批处理优化
uv run python app_interpolation_uns.py -g 48 48 48 -b 15000 -o standard_result
```

### 示例3：高质量分析
```bash
# 64×64×64网格，最高精度
uv run python app_interpolation_uns.py -g 64 64 64 -o high_quality
```

### 示例4：特殊配置
```bash
# 32×32×32网格，禁用SDF，最近邻插值
uv run python app_interpolation_uns.py -g 32 32 32 --method nearest --no-sdf -o full_domain
```

## 输出字段说明

生成的HDF5文件包含以下物理场：

- **Velocity**: 速度场 (3D矢量)
- **P**: 压力场 (标量)
- **RHO**: 密度场 (标量)
- **NodeID**: 节点ID (标量)
- **SDF**: 符号距离场 (标量，仅当启用SDF时)

## 性能优化建议

1. **批处理大小**:
   - 内存充足时：增大`--batch-size` (15000-20000)
   - 内存受限时：减小`--batch-size` (5000-8000)

2. **网格选择**:
   - 测试阶段：使用16×16×16或32×32×32
   - 生产环境：根据精度需求选择48×48×48或64×64×64

3. **SDF计算**:
   - 如果需要全域插值：使用`--no-sdf`参数
   - 仅血管内部插值：保持默认SDF启用

## 故障排除

### 常见问题

1. **内存不足错误**:
   ```bash
   # 减小批处理大小
   uv run python app_interpolation_uns.py -g 32 32 32 -b 5000
   ```

2. **文件路径错误**:
   - 确保在`app/`目录下运行
   - 检查`data_UNS/`目录是否存在

3. **权限错误**:
   - 确保对`data_matrix/`目录有写权限

### 调试模式

如需详细调试信息，可以修改代码中的日志级别。

## 技术架构

- **数据读取**: `VTKReader` - VTU/VTM格式支持
- **几何处理**: `STLReader` + `SDF_Utils` - 符号距离场计算
- **插值算法**: `OptimizedGridInterpolator` - 优化版插值器
- **数据存储**: `HDF5Storage` - 高效数据存储和VTK转换

## 更新日志

### v1.0.0
- 基于`test_32x32x32_quick_verification.py`开发
- 支持用户可配置网格大小
- 添加完整的命令行参数支持
- 集成性能统计和安全检查
- 完善的错误处理和用户提示

**文件更新**:
- 主程序重命名为 `app_interpolation_uns.py`

## 联系支持

如有问题或建议，请：
1. 检查本README的故障排除部分
2. 查看项目文档`../docs/`目录
3. 提交Issue到项目仓库