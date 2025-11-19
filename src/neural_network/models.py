"""
神经网络模型定义
实现从SDF到速度场的映射网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SDFToVelocityNet(nn.Module):
    """
    从SDF一维展开到速度场一维展开的神经网络

    输入: SDF展平向量 [batch_size, grid_points]
    输出: 速度场展平向量 [batch_size, grid_points * 3]
    """

    def __init__(self,
                 grid_size: Tuple[int, int, int] = (64, 64, 64),
                 hidden_dims: list = [1024, 2048, 4096, 2048, 1024],
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True,
                 activation: str = 'relu'):
        super().__init__()

        self.grid_size = grid_size
        self.grid_points = np.prod(grid_size)  # 64*64*64 = 262144
        self.output_dim = self.grid_points * 3  # 速度场3个分量

        # 验证输入维度
        if len(hidden_dims) < 2:
            raise ValueError("hidden_dims should have at least 2 layers")

        # 构建网络层
        layers = []

        # 输入层
        layers.append(nn.Linear(self.grid_points, hidden_dims[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(self._get_activation(activation))
        layers.append(nn.Dropout(dropout_rate))

        # 隐藏层
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout_rate))

        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], self.output_dim))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'swish':
            return nn.SiLU()
        else:
            return nn.ReLU(inplace=True)

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, sdf_flat: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            sdf_flat: [batch_size, grid_points] SDF一维向量

        Returns:
            velocity_flat: [batch_size, grid_points * 3] 速度场一维向量
        """
        # 输入验证
        if sdf_flat.dim() != 2:
            raise ValueError(f"Expected 2D input, got {sdf_flat.dim()}D")
        if sdf_flat.size(1) != self.grid_points:
            raise ValueError(f"Expected input size {self.grid_points}, got {sdf_flat.size(1)}")

        # 前向传播
        velocity_flat = self.network(sdf_flat)

        return velocity_flat

    def get_output_shape(self, batch_size: int) -> Tuple[int, ...]:
        """获取输出形状"""
        return (batch_size, self.output_dim)


class ConvolutionalSDFNet(nn.Module):
    """
    基于卷积的SDF到速度场映射网络
    先将1D SDF重塑为3D，使用3D卷积处理，再展平输出
    """

    def __init__(self,
                 grid_size: Tuple[int, int, int] = (64, 64, 64),
                 base_channels: int = 32,
                 num_layers: int = 4,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.grid_size = grid_size
        self.base_channels = base_channels
        self.output_dim = np.prod(grid_size) * 3

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

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels * 2),
            nn.ReLU(inplace=True)
        )

        # 解码器
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            out_channels = base_channels * (2 ** i)
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose3d(in_channels * 2 if i == num_layers - 1 else in_channels,
                                 out_channels, kernel_size=2, stride=2),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout_rate)
            ))
            in_channels = out_channels

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv3d(in_channels, 3, kernel_size=1),  # 3个速度分量
            nn.Tanh()  # 限制输出范围
        )

        # 池化层用于下采样
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, sdf_flat: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            sdf_flat: [batch_size, grid_points] SDF一维向量

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

        # 瓶颈层
        x = self.bottleneck(x)

        # 解码路径
        for i, decoder_layer in enumerate(self.decoder_layers):
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


class HybridSDFNet(nn.Module):
    """
    混合架构网络：结合全连接和卷积操作
    """

    def __init__(self,
                 grid_size: Tuple[int, int, int] = (64, 64, 64),
                 fc_dims: list = [1024, 2048],
                 conv_channels: int = 32,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.grid_size = grid_size
        self.grid_points = np.prod(grid_size)

        # 全连接分支
        fc_layers = []
        in_dim = self.grid_points

        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(in_dim, fc_dim),
                nn.BatchNorm1d(fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            in_dim = fc_dim

        self.fc_branch = nn.Sequential(*fc_layers)

        # 重塑为3D用于卷积处理
        self.reshape_size = (
            conv_channels,
            grid_size[0] // 4,  # 假设我们缩小空间维度
            grid_size[1] // 4,
            grid_size[2] // 4
        )

        # 计算重塑后的维度
        reshape_dim = np.prod(self.reshape_size)

        # 连接层
        self.connector = nn.Linear(fc_dims[-1], reshape_dim)

        # 卷积分支
        self.conv_branch = nn.Sequential(
            nn.Conv3d(self.reshape_size[0], self.reshape_size[0] * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.reshape_size[0] * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.reshape_size[0] * 2, self.reshape_size[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(self.reshape_size[0]),
            nn.ReLU(inplace=True)
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv3d(self.reshape_size[0], 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 3, kernel_size=1),
            nn.Tanh()
        )

        # 上采样到原始尺寸
        self.upsampler = nn.Upsample(size=grid_size, mode='trilinear', align_corners=False)

    def forward(self, sdf_flat: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        batch_size = sdf_flat.size(0)

        # 全连接分支
        fc_features = self.fc_branch(sdf_flat)

        # 连接和重塑
        connected = self.connector(fc_features)
        conv_input = connected.view(batch_size, *self.reshape_size)

        # 卷积分支
        conv_features = self.conv_branch(conv_input)

        # 输出层
        velocity_3d = self.output_layer(conv_features)

        # 上采样到原始尺寸
        velocity_3d = self.upsampler(velocity_3d)

        # 展平
        velocity_flat = velocity_3d.view(batch_size, -1)

        return velocity_flat


def create_model(model_type: str = 'fc', **kwargs) -> nn.Module:
    """
    创建模型的工厂函数

    Args:
        model_type: 模型类型 ('fc', 'conv', 'hybrid')
        **kwargs: 模型参数

    Returns:
        model: 创建的神经网络模型
    """
    if model_type.lower() == 'fc':
        return SDFToVelocityNet(**kwargs)
    elif model_type.lower() == 'conv':
        return ConvolutionalSDFNet(**kwargs)
    elif model_type.lower() == 'hybrid':
        return HybridSDFNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: 'fc', 'conv', 'hybrid'")


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建测试数据
    batch_size = 2
    grid_size = (64, 64, 64)
    grid_points = np.prod(grid_size)

    sdf_input = torch.randn(batch_size, grid_points).to(device)

    # 测试全连接模型
    print("\n=== 测试全连接模型 ===")
    fc_model = SDFToVelocityNet(grid_size=grid_size).to(device)
    print(f"FC Model parameters: {sum(p.numel() for p in fc_model.parameters()):,}")

    with torch.no_grad():
        fc_output = fc_model(sdf_input)
        print(f"FC Input shape: {sdf_input.shape}")
        print(f"FC Output shape: {fc_output.shape}")
        print(f"Expected output shape: ({batch_size}, {grid_points * 3})")

    # 测试卷积模型
    print("\n=== 测试卷积模型 ===")
    conv_model = ConvolutionalSDFNet(grid_size=grid_size).to(device)
    print(f"Conv Model parameters: {sum(p.numel() for p in conv_model.parameters()):,}")

    with torch.no_grad():
        conv_output = conv_model(sdf_input)
        print(f"Conv Input shape: {sdf_input.shape}")
        print(f"Conv Output shape: {conv_output.shape}")

    # 测试混合模型
    print("\n=== 测试混合模型 ===")
    hybrid_model = HybridSDFNet(grid_size=grid_size).to(device)
    print(f"Hybrid Model parameters: {sum(p.numel() for p in hybrid_model.parameters()):,}")

    with torch.no_grad():
        hybrid_output = hybrid_model(sdf_input)
        print(f"Hybrid Input shape: {sdf_input.shape}")
        print(f"Hybrid Output shape: {hybrid_output.shape}")

    print("\n模型测试完成！")