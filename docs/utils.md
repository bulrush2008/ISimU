# 通用工具模块文档

## 模块概述
通用工具模块(`utils`)是ISimU项目的支撑模块，提供跨模块的通用功能和服务。该模块包含可视化工具、配置管理、IO操作、性能监控等功能，为数据处理、神经网络训练和推理预测等核心模块提供基础支撑。

## 模块功能

### 核心功能
- **可视化工具**：2D/3D流场可视化、图表生成
- **配置管理**：统一的项目配置和参数管理
- **IO工具**：文件操作、数据格式转换、路径管理
- **性能监控**：计算时间监控、内存使用统计
- **日志系统**：统一的日志记录和管理
- **调试工具**：开发辅助和调试功能

## 技术实现

### 核心技术栈
- **可视化**：Matplotlib, Plotly, Mayavi, VTK
- **配置管理**：YAML, JSON, argparse
- **文件处理**：pathlib, shutil, gzip
- **性能监控**：time, psutil, memory_profiler
- **日志系统**：logging, rich, tqdm

## 模块组件

### 文件结构
```
src/utils/
├── __init__.py                 # 模块入口
├── visualization.py            # 可视化工具
├── config.py                  # 配置管理
├── io_utils.py                # IO工具
├── performance.py             # 性能监控
├── logger.py                  # 日志系统
├── debug.py                   # 调试工具
└── decorators.py              # 装饰器工具
```

### 主要组件说明

#### visualization.py - 可视化工具
```python
class FlowVisualizer:
    """流场可视化器"""

    def plot_pressure_field(self, pressure, coords, slice_axis='z', slice_idx=None):
        """绘制压力场切片"""

    def plot_velocity_field(self, velocity, coords, mode='quiver'):
        """绘制速度场"""

    def plot_streamlines(self, velocity, coords, seed_points=None):
        """绘制流线"""

    def create_3d_animation(self, data_sequence, output_file):
        """创建3D动画"""

    def export_to_paraview(self, data, filename):
        """导出ParaView兼容格式"""

class ComparisonPlotter:
    """对比可视化工具"""

    def compare_fields(self, field1, field2, title1, title2):
        """对比两个物理场"""

    def plot_error_distribution(self, pred, target):
        """绘制误差分布"""
```

#### config.py - 配置管理
```python
class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path=None):
        """初始化配置管理器"""

    def load_config(self, config_path):
        """加载配置文件"""

    def save_config(self, config_path):
        """保存配置文件"""

    def get(self, key, default=None):
        """获取配置项"""

    def set(self, key, value):
        """设置配置项"""

    def update(self, updates):
        """批量更新配置"""

class ExperimentConfig:
    """实验配置类"""

    def __init__(self, **kwargs):
        """初始化实验配置"""

    def to_dict(self):
        """转换为字典"""

    def from_dict(self, config_dict):
        """从字典加载配置"""

    def validate(self):
        """验证配置有效性"""
```

#### io_utils.py - IO工具
```python
class FileManager:
    """文件管理器"""

    def ensure_dir(self, path):
        """确保目录存在"""

    def backup_file(self, file_path, backup_suffix='_backup'):
        """备份文件"""

    def clean_directory(self, directory, pattern='*'):
        """清理目录"""

    def get_file_size(self, file_path):
        """获取文件大小"""

    def find_files(self, directory, pattern, recursive=True):
        """查找文件"""

class DataConverter:
    """数据格式转换器"""

    def vtk_to_hdf5(self, vtk_file, hdf5_file):
        """VTK转HDF5"""

    def hdf5_to_numpy(self, hdf5_file, field_name):
        """HDF5转NumPy数组"""

    def numpy_to_vtk(self, array, grid_info):
        """NumPy数组转VTK"""

class PathManager:
    """路径管理器"""

    def __init__(self, base_dir):
        """初始化路径管理器"""

    def get_data_path(self, *args):
        """获取数据路径"""

    def get_model_path(self, *args):
        """获取模型路径"""

    def get_result_path(self, *args):
        """获取结果路径"""
```

#### performance.py - 性能监控
```python
class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        """初始化性能监控器"""

    def start_timer(self, name):
        """开始计时"""

    def end_timer(self, name):
        """结束计时"""

    def get_elapsed_time(self, name):
        """获取经过时间"""

    def get_memory_usage(self):
        """获取内存使用情况"""

    def log_performance_stats(self):
        """记录性能统计"""

class BenchmarkRunner:
    """基准测试运行器"""

    def run_benchmark(self, func, *args, **kwargs):
        """运行基准测试"""

    def compare_functions(self, funcs, inputs):
        """比较函数性能"""

    def generate_report(self, results):
        """生成性能报告"""

class ResourceMonitor:
    """资源监控器"""

    def monitor_memory(self, interval=1.0):
        """监控内存使用"""

    def monitor_gpu(self, interval=1.0):
        """监控GPU使用"""

    def get_system_info(self):
        """获取系统信息"""
```

#### logger.py - 日志系统
```python
class Logger:
    """统一日志管理器"""

    def __init__(self, name='isimu', level='INFO'):
        """初始化日志器"""

    def info(self, message, **kwargs):
        """信息日志"""

    def warning(self, message, **kwargs):
        """警告日志"""

    def error(self, message, **kwargs):
        """错误日志"""

    def debug(self, message, **kwargs):
        """调试日志"""

    def log_progress(self, iterable, desc=None, total=None):
        """记录进度"""

class ExperimentLogger:
    """实验日志记录器"""

    def __init__(self, experiment_dir):
        """初始化实验日志"""

    def log_config(self, config):
        """记录实验配置"""

    def log_metrics(self, metrics, step):
        """记录评估指标"""

    def log_model_info(self, model):
        """记录模型信息"""

    def save_logs(self):
        """保存日志"""
```

#### debug.py - 调试工具
```python
class Debugger:
    """调试工具集"""

    def print_shape(self, array, name=None):
        """打印数组形状"""

    def print_memory_usage(self):
        """打印内存使用"""

    def trace_function(self, func):
        """函数调用追踪装饰器"""

    def debug_array(self, array, name=None, stats=True):
        """调试数组信息"""

class ProfilerManager:
    """性能分析管理器"""

    def start_profiling(self, name):
        """开始性能分析"""

    def stop_profiling(self, name):
        """停止性能分析"""

    def get_profile_stats(self, name):
        """获取性能分析结果"""

class DataValidator:
    """数据验证工具"""

    def validate_hdf5(self, file_path):
        """验证HDF5文件"""

    def validate_vtk(self, file_path):
        """验证VTK文件"""

    def check_data_consistency(self, data1, data2):
        """检查数据一致性"""
```

#### decorators.py - 装饰器工具
```python
def timing_decorator(func):
    """计时装饰器"""

def memory_monitor(func):
    """内存监控装饰器"""

def retry_decorator(max_retries=3, delay=1.0):
    """重试装饰器"""

def cache_decorator(ttl=3600):
    """缓存装饰器"""

def validation_decorator(validation_func):
    """验证装饰器"""
```

## 使用示例

### 可视化工具使用
```python
from utils.visualization import FlowVisualizer, ComparisonPlotter
import matplotlib.pyplot as plt

# 创建可视化器
visualizer = FlowVisualizer()

# 绘制压力场切片
visualizer.plot_pressure_field(
    pressure=pressure_field,
    coords=grid_coordinates,
    slice_axis='z',
    slice_idx=32
)

# 绘制速度场矢量图
visualizer.plot_velocity_field(
    velocity=velocity_field,
    coords=grid_coordinates,
    mode='quiver'
)

# 创建3D流线动画
visualizer.create_3d_animation(
    data_sequence=time_series_data,
    output_file='flow_animation.mp4'
)

# 对比可视化
comparison = ComparisonPlotter()
comparison.compare_fields(
    field1=predicted_pressure,
    field2=ground_truth_pressure,
    title1='Predicted',
    title2='Ground Truth'
)
```

### 配置管理使用
```python
from utils.config import ConfigManager, ExperimentConfig

# 配置管理
config = ConfigManager('configs/default_config.yaml')

# 获取配置
batch_size = config.get('training.batch_size', 32)
learning_rate = config.get('training.learning_rate', 1e-4)

# 更新配置
config.set('training.epochs', 100)
config.update({
    'model.base_features': 64,
    'data.augmentation': True
})

# 实验配置
exp_config = ExperimentConfig(
    model_type='3d_unet',
    grid_size=(64, 64, 64),
    batch_size=16
)

# 保存实验配置
config.save_config('configs/experiment_001.yaml')
```

### 性能监控使用
```python
from utils.performance import PerformanceMonitor, BenchmarkRunner
from time import sleep

# 性能监控
monitor = PerformanceMonitor()

# 监控代码块性能
monitor.start_timer('data_loading')
# ... 数据加载代码 ...
monitor.end_timer('data_loading')

# 监控内存使用
memory_info = monitor.get_memory_usage()
print(f"当前内存使用: {memory_info['used_gb']:.2f} GB")

# 基准测试
benchmark = BenchmarkRunner()

def test_function(size):
    """测试函数"""
    result = np.random.rand(size, size)
    return np.sum(result)

# 运行基准测试
benchmark_result = benchmark.run_benchmark(
    test_function,
    size=1000
)

print(f"函数执行时间: {benchmark_result['time']:.4f}s")
```

### IO工具使用
```python
from utils.io_utils import FileManager, DataConverter, PathManager

# 文件管理
file_manager = FileManager()

# 确保目录存在
file_manager.ensure_dir('results/experiment_001')

# 备份重要文件
file_manager.backup_file('models/best_model.pth')

# 查找所有HDF5文件
hdf5_files = file_manager.find_files('matrix_data/', '*.h5')

# 数据转换
converter = DataConverter()

# VTK转HDF5
converter.vtk_to_hdf5(
    vtk_file='output/flow_field.vts',
    hdf5_file='output/flow_field.h5'
)

# 路径管理
path_manager = PathManager('D:/Devel/ISimU')

# 获取各种路径
data_path = path_manager.get_data_path('vessel.000170.vtm')
model_path = path_manager.get_model_path('best_model.pth')
result_path = path_manager.get_result_path('experiment_001', 'results.h5')
```

### 日志系统使用
```python
from utils.logger import Logger, ExperimentLogger

# 基础日志
logger = Logger('my_experiment')

logger.info("实验开始")
logger.warning("检测到内存使用较高")
logger.error("模型加载失败", error_code=404)

# 进度记录
for i in logger.log_progress(range(100), desc="处理进度"):
    # ... 处理代码 ...
    sleep(0.01)

# 实验日志
exp_logger = ExperimentLogger('experiments/exp_001')

# 记录配置
exp_logger.log_config({
    'model': '3d_unet',
    'batch_size': 32,
    'learning_rate': 1e-4
})

# 记录训练指标
for epoch in range(100):
    # ... 训练代码 ...
    metrics = {'loss': 0.1, 'accuracy': 0.95}
    exp_logger.log_metrics(metrics, epoch)
```

### 调试工具使用
```python
from utils.debug import Debugger, ProfilerManager, DataValidator
from utils.decorators import timing_decorator, memory_monitor

# 调试工具
debugger = Debugger()

# 检查数组信息
debugger.print_shape(pressure_field, name='pressure')
debugger.debug_array(velocity_field, name='velocity', stats=True)

# 性能分析
profiler = ProfilerManager()

profiler.start_profiling('training_loop')
# ... 训练代码 ...
profiler.stop_profiling('training_loop')

# 获取性能统计
stats = profiler.get_profile_stats('training_loop')
print(f"总耗时: {stats['total_time']:.2f}s")

# 数据验证
validator = DataValidator()

# 验证HDF5文件
is_valid = validator.validate_hdf5('matrix_data/output.h5')
if not is_valid:
    print("HDF5文件验证失败")

# 使用装饰器
@timing_decorator
@memory_monitor
def train_epoch(model, data_loader):
    """训练一个epoch"""
    # ... 训练代码 ...
    return loss
```

## 配置文件格式

### 主配置文件 (config.yaml)
```yaml
# 项目配置
project:
  name: "ISimU"
  version: "2.0.0"
  description: "CFD代理模型开发平台"

# 数据处理配置
data_processing:
  grid_size: [64, 64, 64]
  interpolation_method: "linear"
  sdf_batch_size: 15000
  geometry_scale: 0.001

# 神经网络配置
neural_network:
  model:
    type: "3d_unet"
    base_features: 32
    num_layers: 4

  training:
    batch_size: 4
    learning_rate: 1e-4
    epochs: 1000
    optimizer: "Adam"
    scheduler: "CosineAnnealingLR"

# 推理配置
inference:
  device: "cuda"
  batch_size: 8
  precision: "fp32"
  optimization:
    enable_tensorrt: false
    chunk_large_grids: true

# 可视化配置
visualization:
  figure_size: [10, 8]
  colormap: "jet"
  save_format: "png"
  dpi: 300

# 日志配置
logging:
  level: "INFO"
  file: "logs/isimu.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 实验配置文件 (experiment_config.yaml)
```yaml
# 实验元信息
experiment:
  name: "exp_001_baseline"
  description: "基线模型实验"
  start_time: "2025-11-14T10:00:00"
  tags: ["baseline", "64x64x64"]

# 数据配置
data:
  source_files:
    - "matrix_data/dense_64x64x64_zero_assignment.h5"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  augmentation: true

# 模型配置
model:
  architecture: "cfd_proxy_model"
  input_channels: 4
  output_channels: 4
  base_features: 32

# 训练配置
training:
  epochs: 500
  batch_size: 2
  learning_rate: 1e-4
  weight_decay: 1e-5

  # 损失函数权重
  loss_weights:
    pressure: 1.0
    velocity: 1.0
    physics: 0.1

# 评估配置
evaluation:
  metrics:
    - "mae"
    - "rmse"
    - "r2_score"
    - "physical_consistency"

  validation_freq: 10
  save_best_model: true
```

## 开发状态

### 当前阶段：📋 规划中
- [x] 模块结构设计
- [ ] 可视化工具实现
- [ ] 配置管理系统
- [ ] IO工具开发
- [ ] 性能监控系统
- [ ] 日志系统实现

### 开发优先级
1. **高优先级**：配置管理、基础IO工具
2. **中优先级**：可视化工具、日志系统
3. **低优先级**：性能监控、调试工具

### 集成计划
- **第一阶段**：基础工具实现，支持核心模块
- **第二阶段**：可视化增强，实验管理功能
- **第三阶段**：性能优化，高级调试功能

## 最佳实践

### 代码组织
- 使用装饰器简化重复代码
- 统一的错误处理和日志记录
- 模块化的配置管理

### 性能考虑
- 合理使用缓存机制
- 内存高效的实现
- GPU加速的可视化

### 可维护性
- 详细的文档字符串
- 单元测试覆盖
- 版本兼容性考虑

---

*模块版本：v0.1 - 规划中*
*最后更新：2025-11-14*
*维护者：ISimU开发团队*