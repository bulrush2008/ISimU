"""
神经网络模型定义

包含多种用于SDF到速度场映射的神经网络架构：
- ConvolutionalSDFNet: 基于CNN的U-Net架构，支持边界条件输入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
import warnings


class ConvolutionalSDFNet(nn.Module):
    """
    基于CNN的SDF到速度场映射网络（支持边界条件输入）
    使用3D U-Net架构，在瓶颈层融合边界条件
    """

    def __init__(self,
                 grid_size: Tuple[int, int, int] = (64, 64, 64),
                 base_channels: int = 32,
                 num_layers: int = 4,
                 dropout_rate: float = 0.1,
                 bc_dim: int = 10):
        super().__init__()

        self.grid_size = grid_size
        self.base_channels = base_channels
        self.output_dim = np.prod(grid_size) * 3
        self.bc_dim = bc_dim
        self.num_layers = num_layers

        # 计算瓶颈层空间尺寸
        self.bottleneck_spatial_size = [dim // (2 ** num_layers) for dim in grid_size]
        self.bottleneck_channels = base_channels * (2 ** (num_layers - 1))

        # 编码器
        self.encoder_layers = nn.ModuleList()
        in_channels = 1  # SDF作为单通道输入

        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            self.encoder_layers.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout_rate)
            ))
            in_channels = out_channels

        # 瓶颈层（处理SDF特征）
        self.sdf_bottleneck = nn.Sequential(
            nn.Conv3d(in_channels, self.bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.bottleneck_channels),
            nn.ReLU(inplace=True)
        )

        # 边界条件处理网络
        self.bc_processor = nn.Sequential(
            nn.Linear(bc_dim, self.bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(self.bottleneck_channels, self.bottleneck_channels),
            nn.ReLU(inplace=True)
        )

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Conv3d(self.bottleneck_channels * 2, self.bottleneck_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.bottleneck_channels * 2),
            nn.ReLU(inplace=True)
        )

        # 解码器 - 需要处理两种情况：有边界条件和无边界条件
        # 存储解码器参数，稍后根据实际情况构建
        self.decoder_params = {
            'num_layers': num_layers,
            'base_channels': base_channels,
            'dropout_rate': dropout_rate
        }

        # 创建两个解码器：一个用于有边界条件，一个用于无边界条件
        self.decoder_with_bc, final_channels_with_bc = self._create_decoder(
            self.bottleneck_channels * 2, base_channels, num_layers, dropout_rate
        )
        self.decoder_without_bc, final_channels_without_bc = self._create_decoder(
            self.bottleneck_channels, base_channels, num_layers, dropout_rate
        )

        # 确保两个解码器的最终输出通道数相同
        final_channels = base_channels
        self.output_layer = nn.Sequential(
            nn.Conv3d(final_channels, 3, kernel_size=1),  # 3个速度分量
            nn.Tanh()  # 限制输出范围
        )

        # 池化层用于下采样
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # 打印模型参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ConvolutionalSDFNet参数量: {total_params:,} (~{total_params / 1e6:.2f}M)")

    def _create_decoder(self, in_channels: int, base_channels: int, num_layers: int, dropout_rate: float) -> Tuple[nn.ModuleList, int]:
        """创建解码器"""
        decoder_layers = nn.ModuleList()
        current_channels = in_channels

        for i in range(num_layers - 1, -1, -1):
            out_channels = base_channels * (2 ** i)
            decoder_layers.append(nn.Sequential(
                nn.ConvTranspose3d(current_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout_rate)
            ))
            current_channels = out_channels

        return decoder_layers, current_channels

    def forward(self, sdf_flat: torch.Tensor, bc_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            sdf_flat: [batch_size, grid_points] SDF一维向量
            bc_params: [batch_size, bc_dim] 边界条件参数（可选）

        Returns:
            velocity_flat: [batch_size, grid_points * 3] 速度场一维向量
        """
        batch_size = sdf_flat.size(0)

        # 重塑为3D: [batch_size, 1, D, H, W]
        sdf_3d = sdf_flat.view(batch_size, 1, *self.grid_size)

        # 编码路径
        encoder_features = []
        x = sdf_3d

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encoder_features.append(x)
            x = self.pool(x)

        # SDF瓶颈层处理
        sdf_features = self.sdf_bottleneck(x)  # [batch_size, bottleneck_channels, d, h, w]

        # 处理边界条件
        if bc_params is not None:
            # 边界条件通过全连接网络处理
            bc_features = self.bc_processor(bc_params)  # [batch_size, bottleneck_channels]

            # 将BC特征扩展到3D空间以匹配SDF特征
            bc_features_3d = bc_features.view(
                batch_size, self.bottleneck_channels, 1, 1, 1
            ).expand(
                batch_size,
                self.bottleneck_channels,
                *self.bottleneck_spatial_size
            )  # [batch_size, bottleneck_channels, d, h, w]

            # 特征融合
            combined_features = torch.cat([sdf_features, bc_features_3d], dim=1)  # [batch_size, bottleneck_channels*2, d, h, w]
            fused_features = self.feature_fusion(combined_features)
            decoder_input_channels = self.bottleneck_channels * 2
        else:
            # 如果没有边界条件，直接使用SDF特征
            if self.training:
                # 训练时建议提供边界条件
                warnings.warn("No boundary conditions provided during training. Consider using bc_params.")

            fused_features = sdf_features
            decoder_input_channels = self.bottleneck_channels

        # 选择合适的解码器
        if bc_params is not None:
            decoder_layers = self.decoder_with_bc
        else:
            decoder_layers = self.decoder_without_bc

        # 解码路径
        x = fused_features
        for i, decoder_layer in enumerate(decoder_layers):
            # 跳跃连接
            if i < len(encoder_features):
                skip = encoder_features[-(i + 1)]
                # 上采样到匹配尺寸
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)

            x = decoder_layer(x)

        # 输出层
        velocity_3d = self.output_layer(x)  # [batch_size, 3, D, H, W]

        # 展平输出
        velocity_flat = velocity_3d.view(batch_size, -1)

        return velocity_flat

    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_type': 'ConvolutionalSDFNet',
            'input_dim': np.prod(self.grid_size),
            'output_dim': self.output_dim,
            'grid_size': self.grid_size,
            'bc_dim': self.bc_dim,
            'num_layers': self.num_layers,
            'base_channels': self.base_channels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_usage_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'supports_boundary_conditions': True
        }

    def get_output_shape(self, batch_size: int) -> tuple:
        """获取输出形状"""
        return (batch_size, self.output_dim)


# 工厂函数
def create_model(model_type: str = 'cnn',
                 grid_size: Tuple[int, int, int] = (64, 64, 64),
                 **kwargs) -> nn.Module:
    """
    创建模型的工厂函数

    Args:
        model_type: 模型类型 ('cnn')
        grid_size: 网格尺寸
        **kwargs: 模型特定参数

    Returns:
        model: 神经网络模型
    """
    if model_type.lower() == 'cnn':
        return ConvolutionalSDFNet(grid_size=grid_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 测试模型创建
    print("=== 测试神经网络模型 ===")

    # 测试CNN模型（无边界条件）
    print("\n1. 测试CNN模型（无边界条件）:")
    cnn_model = ConvolutionalSDFNet(
        grid_size=(16, 16, 16),  # 使用更小网格进行测试
        base_channels=8,
        num_layers=2,
        bc_dim=4
    )

    # 创建测试输入
    batch_size = 1
    sdf_input = torch.randn(batch_size, 16*16*16)
    bc_input = torch.randn(batch_size, 4)

    # 前向传播测试
    with torch.no_grad():
        # 测试有边界条件（应该工作正常）
        output_with_bc = cnn_model(sdf_input, bc_input)
        print(f"  输入SDF形状: {sdf_input.shape}")
        print(f"  边界条件形状: {bc_input.shape}")
        print(f"  输出形状（有BC）: {output_with_bc.shape}")

        # 测试无边界条件
        cnn_model.eval()  # 避免警告
        output_no_bc = cnn_model(sdf_input)
        print(f"  输出形状（无BC）: {output_no_bc.shape}")

    # 模型信息
    model_info = cnn_model.get_model_info()
    print(f"\n模型信息:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # 测试工厂函数
    print("\n2. 测试工厂函数:")
    factory_model = create_model(
        model_type='cnn',
        grid_size=(32, 32, 32),
        base_channels=8,
        num_layers=2,
        bc_dim=5
    )

    factory_info = factory_model.get_model_info()
    print(f"  工厂模型类型: {factory_info['model_type']}")
    print(f"  参数量: {factory_info['total_parameters']:,}")

    print("\n✅ 神经网络模型测试通过！")