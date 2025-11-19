"""
损失函数定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any


class MSELossWithMask(nn.Module):
    """
    带掩码的均方误差损失
    只计算血管内部区域的损失
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor, sdf: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算带掩码的MSE损失

        Args:
            pred: 预测值 [batch_size, grid_points * 3]
            target: 目标值 [batch_size, grid_points * 3]
            sdf: 符号距离场 [batch_size, grid_points], 用于创建掩码

        Returns:
            loss: 计算的损失值
        """
        # 计算基础MSE损失
        loss = self.mse_loss(pred, target)

        # 如果提供了SDF，创建掩码
        if sdf is not None:
            # SDF > 0 表示血管内部
            mask = (sdf > 0).float()  # [batch_size, grid_points]
            # 扩展掩码到3个速度分量
            mask = mask.unsqueeze(1).expand(-1, 3, -1)  # [batch_size, 3, grid_points]
            mask = mask.reshape(pred.shape)  # [batch_size, grid_points * 3]

            # 应用掩码
            loss = loss * mask

            # 计算有效点的数量
            valid_points = mask.sum()
            if valid_points > 0:
                loss = loss.sum() / valid_points
            else:
                loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        else:
            # 没有掩码时使用标准MSE
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        return loss


class WeightedMSELoss(nn.Module):
    """
    加权均方误差损失
    对血管内部和外部区域使用不同权重
    """

    def __init__(self, inside_weight: float = 1.0, outside_weight: float = 0.1):
        super().__init__()
        self.inside_weight = inside_weight
        self.outside_weight = outside_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        """
        计算加权MSE损失

        Args:
            pred: 预测值 [batch_size, grid_points * 3]
            target: 目标值 [batch_size, grid_points * 3]
            sdf: 符号距离场 [batch_size, grid_points]

        Returns:
            loss: 计算的损失值
        """
        # 重塑预测和目标
        batch_size = pred.size(0)
        pred_reshaped = pred.view(batch_size, -1, 3)  # [batch_size, grid_points, 3]
        target_reshaped = target.view(batch_size, -1, 3)

        # 计算每个点的MSE
        mse_per_point = ((pred_reshaped - target_reshaped) ** 2).mean(dim=2)  # [batch_size, grid_points]

        # 创建权重
        weights = torch.where(sdf > 0,
                             torch.tensor(self.inside_weight, device=sdf.device),
                             torch.tensor(self.outside_weight, device=sdf.device))

        # 应用权重
        weighted_loss = mse_per_point * weights

        return weighted_loss.mean()


class PhysicsInformedLoss(nn.Module):
    """
    物理信息损失函数
    包含质量守恒（散度为零）等物理约束
    """

    def __init__(self, physics_weight: float = 0.1, divergence_weight: float = 1.0):
        super().__init__()
        self.physics_weight = physics_weight
        self.divergence_weight = divergence_weight

    def compute_divergence(self, velocity: torch.Tensor, grid_size: tuple) -> torch.Tensor:
        """
        计算速度场的散度
        使用有限差分方法

        Args:
            velocity: 速度场 [batch_size, grid_points * 3]
            grid_size: 网格尺寸 (D, H, W)

        Returns:
            divergence: 散度场 [batch_size, grid_points]
        """
        batch_size = velocity.size(0)
        D, H, W = grid_size

        # 重塑速度场
        velocity = velocity.view(batch_size, D, H, W, 3)
        vx = velocity[..., 0]  # [batch_size, D, H, W]
        vy = velocity[..., 1]
        vz = velocity[..., 2]

        # 计算散度 (使用中心差分)
        # X方向导数
        dvx_dx = torch.zeros_like(vx)
        dvx_dx[:, :, :, 1:-1] = (vx[:, :, :, 2:] - vx[:, :, :, :-2]) / 2.0

        # Y方向导数
        dvy_dy = torch.zeros_like(vy)
        dvy_dy[:, :, 1:-1, :] = (vy[:, :, 2:, :] - vy[:, :, :-2, :]) / 2.0

        # Z方向导数
        dvz_dz = torch.zeros_like(vz)
        dvz_dz[:, 1:-1, :, :] = (vz[:, 2:, :, :] - vz[:, :-2, :, :]) / 2.0

        # 散度
        divergence = dvx_dx + dvy_dy + dvz_dz  # [batch_size, D, H, W]
        divergence = divergence.view(batch_size, -1)  # [batch_size, grid_points]

        return divergence

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                sdf: torch.Tensor, grid_size: tuple) -> Dict[str, torch.Tensor]:
        """
        计算物理信息损失

        Args:
            pred: 预测速度场 [batch_size, grid_points * 3]
            target: 目标速度场 [batch_size, grid_points * 3]
            sdf: 符号距离场 [batch_size, grid_points]
            grid_size: 网格尺寸

        Returns:
            losses: 包含各项损失的字典
        """
        batch_size = pred.size(0)

        # 基础MSE损失
        mse_loss = F.mse_loss(pred, target)

        # 散度损失（质量守恒）
        divergence = self.compute_divergence(pred, grid_size)

        # 只在血管内部计算散度损失
        inside_mask = (sdf > 0).float()
        divergence_loss = (divergence ** 2) * inside_mask
        divergence_loss = divergence_loss.sum() / (inside_mask.sum() + 1e-8)

        # 边界条件损失（血管外部速度应该接近零）
        outside_mask = (sdf <= 0).float()
        pred_reshaped = pred.view(batch_size, -1, 3)
        boundary_loss = (pred_reshaped ** 2).mean(dim=2) * outside_mask
        boundary_loss = boundary_loss.sum() / (outside_mask.sum() + 1e-8)

        # 总损失
        total_loss = mse_loss + self.physics_weight * (
            self.divergence_weight * divergence_loss + boundary_loss
        )

        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'divergence_loss': divergence_loss,
            'boundary_loss': boundary_loss
        }


class CombinedLoss(nn.Module):
    """
    组合损失函数
    结合多种损失项
    """

    def __init__(self,
                 mse_weight: float = 1.0,
                 physics_weight: float = 0.1,
                 inside_weight: float = 1.0,
                 outside_weight: float = 0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight

        self.mse_loss = MSELossWithMask()
        self.physics_loss = PhysicsInformedLoss(physics_weight=1.0)
        self.weighted_mse = WeightedMSELoss(inside_weight, outside_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                sdf: torch.Tensor, grid_size: tuple) -> Dict[str, torch.Tensor]:
        """
        计算组合损失

        Args:
            pred: 预测值 [batch_size, grid_points * 3]
            target: 目标值 [batch_size, grid_points * 3]
            sdf: 符号距离场 [batch_size, grid_points]
            grid_size: 网格尺寸

        Returns:
            losses: 包含各项损失的字典
        """
        # 基础MSE损失（仅血管内部）
        mse_inside = self.mse_loss(pred, target, sdf)

        # 加权MSE损失
        weighted_mse = self.weighted_mse(pred, target, sdf)

        # 物理信息损失
        physics_losses = self.physics_loss(pred, target, sdf, grid_size)

        # 组合损失
        total_loss = (
            self.mse_weight * mse_inside +
            self.physics_weight * physics_losses['total_loss']
        )

        return {
            'total_loss': total_loss,
            'mse_inside': mse_inside,
            'weighted_mse': weighted_mse,
            'physics_losses': physics_losses,
            'mse_weight': self.mse_weight,
            'physics_weight': self.physics_weight
        }


def create_loss_function(loss_type: str = 'combined', **kwargs) -> nn.Module:
    """
    创建损失函数的工厂函数

    Args:
        loss_type: 损失函数类型 ('mse', 'weighted_mse', 'physics_informed', 'combined')
        **kwargs: 损失函数参数

    Returns:
        loss_fn: 损失函数
    """
    if loss_type.lower() == 'mse':
        return MSELossWithMask(**kwargs)
    elif loss_type.lower() == 'weighted_mse':
        return WeightedMSELoss(**kwargs)
    elif loss_type.lower() == 'physics_informed':
        return PhysicsInformedLoss(**kwargs)
    elif loss_type.lower() == 'combined':
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # 测试损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing loss functions on {device}")

    # 创建测试数据
    batch_size = 2
    grid_size = (32, 32, 32)  # 使用较小的网格进行测试
    grid_points = np.prod(grid_size)

    pred = torch.randn(batch_size, grid_points * 3).to(device)
    target = torch.randn(batch_size, grid_points * 3).to(device)
    sdf = torch.randn(batch_size, grid_points).to(device)

    print(f"Input shapes: pred={pred.shape}, target={target.shape}, sdf={sdf.shape}")

    # 测试各种损失函数
    loss_functions = {
        'MSE with Mask': MSELossWithMask(),
        'Weighted MSE': WeightedMSELoss(),
        'Physics Informed': PhysicsInformedLoss(),
        'Combined': CombinedLoss()
    }

    for name, loss_fn in loss_functions.items():
        try:
            if name == 'Physics Informed' or name == 'Combined':
                loss_output = loss_fn(pred, target, sdf, grid_size)
                if isinstance(loss_output, dict):
                    print(f"{name}: total_loss={loss_output['total_loss']:.6f}")
                    for key, value in loss_output.items():
                        if key != 'total_loss':
                            print(f"  {key}: {value:.6f}")
                else:
                    print(f"{name}: {loss_output:.6f}")
            else:
                loss = loss_fn(pred, target, sdf)
                print(f"{name}: {loss:.6f}")
        except Exception as e:
            print(f"{name}: Error - {e}")

    print("Loss function testing completed!")