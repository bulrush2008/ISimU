"""
神经网络训练示例
演示如何使用构建的神经网络模块训练CFD代理模型
"""

import sys
import os
import torch
import warnings
from pathlib import Path

# 添加项目根目录到Python路径
base_dir = Path(__file__).parent.parent.parent
src_dir = base_dir / 'src'
sys.path.insert(0, str(src_dir))

from neural_network import (
    CFDDataModule,
    create_trainer,
    create_model
)

def main():
    """主训练函数"""
    print("=== ISimU 神经网络训练示例 ===\n")

    # 配置参数
    config = {
        # 数据配置
        'data_dir': str(base_dir / 'matrix_data'),
        'grid_size': (64, 64, 64),
        'batch_size': 1,  # 根据GPU内存调整
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,

        # 模型配置
        'model_type': 'fc',  # 'fc', 'conv', 'hybrid'
        'model_kwargs': {
            'hidden_dims': [512, 1024, 2048, 1024, 512],  # 减小网络规模适应小数据集
            'dropout_rate': 0.1,
            'use_batch_norm': True,
            'activation': 'relu'
        },

        # 训练配置
        'epochs': 100,  # 快速测试，实际训练可以增加到1000+
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'Adam',
        'scheduler': 'cosine',
        'early_stopping_patience': 20,  # 减小patience用于快速测试

        # 损失函数配置
        'loss_function': 'combined',
        'loss_kwargs': {
            'mse_weight': 1.0,
            'physics_weight': 0.05,  # 降低物理约束权重
            'inside_weight': 1.0,
            'outside_weight': 0.1
        },

        # 其他配置
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_interval': 10,
        'log_interval': 5,
        'gradient_clip_value': 1.0
    }

    print("训练配置:")
    for key, value in config.items():
        if key != 'loss_kwargs':
            print(f"  {key}: {value}")
    print()

    # 检查数据目录
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请确保已经运行了数据处理模块生成了HDF5数据文件")
        return

    # 列出可用的数据文件
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        print(f"❌ 在 {data_dir} 中没有找到HDF5数据文件")
        return

    print(f"✅ 找到 {len(h5_files)} 个HDF5数据文件:")
    for file in h5_files[:5]:  # 只显示前5个
        print(f"  - {file.name}")
    if len(h5_files) > 5:
        print(f"  ... 还有 {len(h5_files) - 5} 个文件")
    print()

    try:
        # 创建数据模块
        print("📊 创建数据模块...")
        data_module = CFDDataModule(
            data_dir=config['data_dir'],
            grid_size=config['grid_size'],
            batch_size=config['batch_size'],
            train_ratio=config['train_ratio'],
            val_ratio=config['val_ratio'],
            test_ratio=config['test_ratio'],
            normalize=True,
            velocity_scale=1.0
        )

        # 获取数据信息
        data_info = data_module.get_data_info()
        print(f"✅ 数据模块创建成功:")
        print(f"  总数据文件: {data_info['total_files']}")
        print(f"  训练集: {data_info['train_files']} 文件")
        print(f"  验证集: {data_info['val_files']} 文件")
        print(f"  测试集: {data_info['test_files']} 文件")
        print(f"  网格尺寸: {data_info['grid_size']}")
        print(f"  输入维度: {data_info['dataset_info']['input_dim']}")
        print(f"  输出维度: {data_info['dataset_info']['output_dim']}")
        print()

        # 检查数据集大小是否足够
        if data_info['train_files'] < 2:
            print("⚠️  警告: 训练数据文件较少，建议生成更多训练数据")
            print("可以通过数据处理模块处理多个VTM文件来生成更多HDF5数据")
            print()

        # 创建训练器
        print("🧠 创建训练器...")
        trainer = create_trainer(
            model_type=config['model_type'],
            data_module=data_module,
            config=config
        )

        # 显示模型信息
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

        print(f"✅ 训练器创建成功:")
        print(f"  模型类型: {config['model_type']}")
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  设备: {trainer.device}")
        print()

        # 开始训练
        print("🚀 开始训练...")
        print("=" * 50)

        # 创建模型保存目录
        models_dir = base_dir / 'models'
        models_dir.mkdir(exist_ok=True)

        # 训练模型
        training_results = trainer.train(save_dir=str(models_dir))

        # 训练完成
        print("=" * 50)
        print("✅ 训练完成!")
        print(f"  总训练轮数: {training_results['total_epochs']}")
        print(f"  最佳验证损失: {training_results['best_val_loss']:.6f}")
        print(f"  训练耗时: {training_results['training_time']/60:.2f} 分钟")
        print(f"  模型保存在: {models_dir}")
        print()

        # 如果有测试集，进行最终评估
        if trainer.test_loader:
            print("🧪 进行最终测试评估...")
            test_results = trainer.evaluate(trainer.test_loader)
            print("✅ 测试结果:")
            print(f"  测试损失: {test_results['total_loss']:.6f}")
            print(f"  测试MAE: {test_results.get('mae_total', 0):.6f}")
            print(f"  测试RMSE: {test_results.get('rmse_total', 0):.6f}")
            print(f"  测试R²: {test_results.get('r2_total', 0):.4f}")
            print(f"  边界违反率: {test_results.get('boundary_violation_ratio', 0):.4f}")
            print()

        # 模型使用示例
        print("📝 模型使用示例:")
        print("```python")
        print("import torch")
        print("from neural_network import create_model")
        print()
        print("# 加载训练好的模型")
        print("model = create_model('fc', grid_size=(64, 64, 64))")
        print(f"checkpoint = torch.load('{models_dir}/best_model.pth')")
        print("model.load_state_dict(checkpoint['model_state_dict'])")
        print("model.eval()")
        print()
        print("# 进行预测")
        print("sdf_input = torch.randn(1, 262144)  # [batch_size, grid_points]")
        print("with torch.no_grad():")
        print("    velocity_pred = model(sdf_input)")
        print("    print(f'预测速度场形状: {velocity_pred.shape}')")
        print("```")

    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎉 神经网络训练示例完成!")


def test_model_architecture():
    """测试不同的模型架构"""
    print("=== 测试模型架构 ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid_size = (64, 64, 64)
    grid_points = 64 * 64 * 64

    # 创建测试输入
    test_input = torch.randn(1, grid_points).to(device)

    model_configs = [
        ('fc', {'hidden_dims': [512, 1024, 2048, 1024, 512]}),
        ('conv', {'base_channels': 16, 'num_layers': 3}),
        ('hybrid', {'fc_dims': [512, 1024], 'conv_channels': 16})
    ]

    for model_type, model_kwargs in model_configs:
        try:
            print(f"测试 {model_type.upper()} 模型...")
            model = create_model(model_type, grid_size=grid_size, **model_kwargs).to(device)

            param_count = sum(p.numel() for p in model.parameters())
            print(f"  参数数量: {param_count:,}")

            with torch.no_grad():
                output = model(test_input)
                print(f"  输入形状: {test_input.shape}")
                print(f"  输出形状: {output.shape}")
                print(f"  ✅ {model_type.upper()} 模型测试成功")

        except Exception as e:
            print(f"  ❌ {model_type.upper()} 模型测试失败: {e}")
        print()


if __name__ == "__main__":
    # 设置警告级别
    warnings.filterwarnings("ignore", category=UserWarning)

    # 可以选择测试模型架构或直接训练
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # 测试模型架构
        test_model_architecture()
    else:
        # 运行训练示例
        main()