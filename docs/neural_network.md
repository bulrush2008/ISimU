# 神经网络模块文档

## 模块概述
神经网络模块(`neural_network`)是ISimU项目的核心AI模块，负责构建、训练和评估用于CFD流场预测的深度学习代理模型。该模块将数据处理模块生成的HDF5矩阵数据作为输入，训练能够快速预测流场的3D卷积神经网络。

## 模块功能

### 核心功能
- **模型架构**：设计和实现3D CNN网络架构
- **数据加载**：高效的HDF5数据加载和预处理
- **模型训练**：完整的训练框架和流程管理
- **模型评估**：多维度的模型性能评估
- **模型管理**：模型的保存、加载和版本管理

### 预期网络架构

#### 3D U-Net架构
- **编码器**：多层3D卷积下采样
- **瓶颈层**：特征提取和压缩
- **解码器**：3D卷积上采样重建
- **跳跃连接**：保留细节信息

#### 注意力机制
- **自注意力层**：增强特征表示能力
- **空间注意力**：关注重要流场区域
- **通道注意力**：优化特征通道权重

#### 残差连接
- **ResNet块**：缓解梯度消失问题
- **密集连接**：促进特征重用

## 技术实现

### 编程语言和框架
- **主语言**：Python 3.8+
- **深度学习框架**：PyTorch 1.9+
- **数值计算**：NumPy, SciPy
- **数据加载**：h5py, pandas
- **可视化**：Matplotlib, TensorBoard

### 网络结构设计

#### 输入层
```python
# 输入数据格式
Input: [B, C, D, H, W]  # Batch, Channels, Depth, Height, Width
# C包含：
# - C=0: SDF (符号距离场)
# - C=1: X坐标
# - C=2: Y坐标
# - C=3: Z坐标
```

#### 输出层
```python
# 输出数据格式
Output: {
    'P': [B, 1, D, H, W],        # 压力场
    'Velocity': [B, 3, D, H, W]   # 速度场 (Vx, Vy, Vz)
}
```

#### 网络层设计
```python
# 示例网络结构
class CFDProxyModel(nn.Module):
    def __init__(self, input_channels=4, output_channels=4, base_features=32):
        # 编码器：4层下采样 (64->32->16->8->4)
        # 瓶颈层：特征提取
        # 解码器：4层上采样 (4->8->16->32->64)
        # 注意力机制：在深层添加
        # 输出层：映射到压力和速度场
```

## 模块组件

### 文件结构
```
src/neural_network/
├── __init__.py                 # 模块入口
├── models.py                   # 网络架构定义
├── datasets.py                 # 数据加载和预处理
├── trainers.py                 # 训练框架
├── losses.py                   # 损失函数定义
├── metrics.py                  # 评估指标
└── utils.py                    # 神经网络工具函数
```

### 主要组件说明

#### models.py - 网络架构
```python
# 主要类
class CFDProxyModel(nn.Module):
    """3D CNN CFD代理模型"""

class SelfAttention(nn.Module):
    """自注意力机制"""

class ResidualBlock3D(nn.Module):
    """3D残差块"""
```

#### datasets.py - 数据加载
```python
# 主要类
class CFDataset(Dataset):
    """CFD数据集加载器"""

class DataAugmentation:
    """数据增强工具"""

class Normalization:
    """数据标准化"""
```

#### trainers.py - 训练框架
```python
# 主要类
class ModelTrainer:
    """模型训练器"""

class TrainingConfig:
    """训练配置"""

class Callback:
    """训练回调函数"""
```

#### losses.py - 损失函数
```python
# 主要损失函数
def physics_informed_loss(pred, target, sdf):
    """物理信息损失函数"""

def weighted_mse_loss(pred, target, sdf):
    """加权MSE损失，考虑血管内外权重"""

def continuity_loss(velocity_field):
    """连续性损失（质量守恒）"""
```

#### metrics.py - 评估指标
```python
# 主要指标
def calculate_mae(pred, target, mask=None):
    """平均绝对误差"""

def calculate_rmse(pred, target, mask=None):
    """均方根误差"""

def calculate_r2_score(pred, target, mask=None):
    """R²决定系数"""

def calculate_velocity_magnitude_error(pred_vel, target_vel, mask=None):
    """速度大小误差"""
```

## 数据处理

### 数据加载流程
1. **HDF5文件读取**：使用`h5py`高效读取
2. **数据预处理**：
   - 坐标标准化
   - 物理量归一化
   - SDF值处理
3. **数据增强**：
   - 随机旋转
   - 翻转变换
   - 噪声添加
4. **批次组织**：动态批处理

### 数据格式标准
```python
# 训练数据格式
{
    'input': {
        'sdf': [B, 1, D, H, W],
        'coordinates': [B, 3, D, H, W]
    },
    'target': {
        'pressure': [B, 1, D, H, W],
        'velocity': [B, 3, D, H, W]
    },
    'metadata': {
        'grid_size': (D, H, W),
        'bounds': (xmin, xmax, ymin, ymax, zmin, zmax),
        'source_file': 'xxx.vtm'
    }
}
```

## 训练配置

### 默认超参数
```python
training_config = {
    # 网络参数
    'input_channels': 4,          # SDF + X + Y + Z
    'output_channels': 4,         # P + Vx + Vy + Vz
    'base_features': 32,
    'num_layers': 4,

    # 训练参数
    'batch_size': 4,              # 根据GPU内存调整
    'learning_rate': 1e-4,
    'num_epochs': 1000,
    'patience': 50,               # 早停耐心值

    # 优化器
    'optimizer': 'Adam',
    'weight_decay': 1e-5,
    'scheduler': 'CosineAnnealingLR',

    # 损失函数权重
    'pressure_weight': 1.0,
    'velocity_weight': 1.0,
    'physics_weight': 0.1,

    # 数据处理
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'augmentation': True,

    # 硬件
    'device': 'cuda',             # 'cuda' 或 'cpu'
    'num_workers': 4,
    'pin_memory': True
}
```

### 训练策略
1. **分阶段训练**：
   - 第一阶段：只训练压力场
   - 第二阶段：联合训练压力场和速度场
   - 第三阶段：物理约束微调

2. **学习率调度**：
   - 预热：前10个epoch线性增加
   - 主体：余弦退火调度
   - 微调：固定小学习率

3. **正则化策略**：
   - Dropout：p=0.1
   - Batch Normalization
   - Weight Decay：1e-5

## 评估指标

### 物理场指标
```python
# 压力场指标
pressure_metrics = {
    'mae': calculate_mae(pred_p, target_p),
    'rmse': calculate_rmse(pred_p, target_p),
    'r2': calculate_r2_score(pred_p, target_p),
    'relative_error': calculate_relative_error(pred_p, target_p)
}

# 速度场指标
velocity_metrics = {
    'mae': calculate_mae(pred_v, target_v),
    'rmse': calculate_rmse(pred_v, target_v),
    'r2': calculate_r2_score(pred_v, target_v),
    'direction_error': calculate_direction_error(pred_v, target_v),
    'magnitude_error': calculate_velocity_magnitude_error(pred_v, target_v)
}
```

### 物理一致性指标
- **质量守恒**： divergence of velocity ≈ 0
- **边界条件**：血管外部速度≈0
- **连续性**：流场连续性检查

### 整体性能指标
- **训练时间**：每epoch耗时
- **推理速度**：单样本预测时间
- **内存使用**：峰值GPU内存占用
- **模型大小**：参数数量和文件大小

## 使用示例

### 基本训练流程
```python
from neural_network import CFDProxyModel, ModelTrainer, CFDataset

# 1. 准备数据
dataset = CFDDataset('matrix_data/', split='train')
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 2. 创建模型
model = CFDProxyModel(input_channels=4, base_features=32)

# 3. 设置训练器
trainer = ModelTrainer(model, config=training_config)

# 4. 开始训练
trainer.train(train_loader, val_loader)

# 5. 评估模型
metrics = trainer.evaluate(test_loader)
print(f"Test MAE: {metrics['mae']:.4f}")
```

### 推理预测
```python
from neural_network import CFDProxyModel
import torch

# 加载训练好的模型
model = CFDProxyModel()
model.load_state_dict(torch.load('models/cfd_proxy_model.pth'))
model.eval()

# 预测
with torch.no_grad():
    input_data = prepare_input_data(hdf5_file)
    output = model(input_data)

    pressure_pred = output['P']
    velocity_pred = output['Velocity']
```

## 性能优化

### 训练加速
1. **混合精度训练**：使用`torch.cuda.amp`
2. **梯度累积**：模拟大批次训练
3. **数据并行**：多GPU训练
4. **分布式训练**：多节点训练

### 内存优化
1. **梯度检查点**：减少激活值内存
2. **数据流水线**：异步数据加载
3. **动态批处理**：根据内存自动调整

### 推理优化
1. **模型量化**：INT8量化加速
2. **模型剪枝**：减少冗余参数
3. **TensorRT优化**：GPU推理加速

## 模型管理

### 版本控制
```python
# 模型保存格式
model_checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'metrics': metrics,
    'config': config,
    'version': '1.0.0'
}
torch.save(model_checkpoint, f'models/model_v{version}.pth')
```

### 实验跟踪
- **TensorBoard**：训练过程可视化
- **Weights & Biases**：实验管理
- **MLflow**：模型生命周期管理

## 开发状态

### 当前阶段：🔄 开发中
- [x] 项目结构创建
- [ ] 3D CNN架构实现
- [ ] 数据加载器开发
- [ ] 训练框架构建
- [ ] 损失函数设计
- [ ] 评估指标实现

### 下一步计划
1. **基础架构实现** (1-2周)
   - 实现3D U-Net基础架构
   - 创建数据加载器
   - 实现基本训练循环

2. **训练框架完善** (1-2周)
   - 添加损失函数
   - 实现评估指标
   - 集成TensorBoard

3. **模型优化** (2-3周)
   - 添加注意力机制
   - 实现物理约束
   - 性能优化和调优

## 开发指南

### 添加新网络架构
1. 在`models.py`中定义新的模型类
2. 继承`nn.Module`基类
3. 实现`forward`方法
4. 添加相应的单元测试

### 自定义损失函数
1. 在`losses.py`中定义新损失函数
2. 确保可微分性
3. 添加梯度计算支持
4. 在训练器中注册使用

### 性能调试
1. 使用`torch.profiler`分析性能瓶颈
2. 检查GPU利用率
3. 优化数据加载流水线
4. 调整批次大小和学习率

---

*模块版本：v0.1 - 开发中*
*最后更新：2025-11-14*
*维护者：ISimU开发团队*