"""
评估指标定义
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings


class CFDMetrics:
    """
    CFD流场预测评估指标
    """

    def __init__(self, grid_size: Tuple[int, int, int]):
        self.grid_size = grid_size
        self.grid_points = np.prod(grid_size)

    def compute_mae(self, pred: torch.Tensor, target: torch.Tensor,
                   sdf: Optional[torch.Tensor] = None, mask_inside: bool = True) -> Dict[str, float]:
        """
        计算平均绝对误差(MAE)

        Args:
            pred: 预测值 [batch_size, grid_points * 3]
            target: 目标值 [batch_size, grid_points * 3]
            sdf: 符号距离场 [batch_size, grid_points]
            mask_inside: 是否只计算血管内部区域

        Returns:
            mae_dict: 包含各项MAE的字典
        """
        batch_size = pred.size(0)

        # 重塑为 [batch_size, grid_points, 3]
        pred_reshaped = pred.view(batch_size, -1, 3)
        target_reshaped = target.view(batch_size, -1, 3)

        # 计算绝对误差
        abs_error = torch.abs(pred_reshaped - target_reshaped)

        # 应用掩码
        if mask_inside and sdf is not None:
            mask = (sdf > 0).float().unsqueeze(-1)  # [batch_size, grid_points, 1]
            abs_error = abs_error * mask
            total_points = mask.sum()
        else:
            total_points = batch_size * self.grid_points

        # 计算各分量的MAE
        mae_x = abs_error[..., 0].sum() / (total_points + 1e-8)
        mae_y = abs_error[..., 1].sum() / (total_points + 1e-8)
        mae_z = abs_error[..., 2].sum() / (total_points + 1e-8)
        mae_total = abs_error.sum() / (total_points + 1e-8)

        return {
            'mae_x': mae_x.item(),
            'mae_y': mae_y.item(),
            'mae_z': mae_z.item(),
            'mae_total': mae_total.item()
        }

    def compute_rmse(self, pred: torch.Tensor, target: torch.Tensor,
                    sdf: Optional[torch.Tensor] = None, mask_inside: bool = True) -> Dict[str, float]:
        """
        计算均方根误差(RMSE)

        Args:
            pred: 预测值 [batch_size, grid_points * 3]
            target: 目标值 [batch_size, grid_points * 3]
            sdf: 符号距离场 [batch_size, grid_points]
            mask_inside: 是否只计算血管内部区域

        Returns:
            rmse_dict: 包含各项RMSE的字典
        """
        batch_size = pred.size(0)

        # 重塑
        pred_reshaped = pred.view(batch_size, -1, 3)
        target_reshaped = target.view(batch_size, -1, 3)

        # 计算平方误差
        squared_error = (pred_reshaped - target_reshaped) ** 2

        # 应用掩码
        if mask_inside and sdf is not None:
            mask = (sdf > 0).float().unsqueeze(-1)
            squared_error = squared_error * mask
            total_points = mask.sum()
        else:
            total_points = batch_size * self.grid_points

        # 计算各分量的RMSE
        rmse_x = torch.sqrt(squared_error[..., 0].sum() / (total_points + 1e-8))
        rmse_y = torch.sqrt(squared_error[..., 1].sum() / (total_points + 1e-8))
        rmse_z = torch.sqrt(squared_error[..., 2].sum() / (total_points + 1e-8))
        rmse_total = torch.sqrt(squared_error.sum() / (total_points + 1e-8))

        return {
            'rmse_x': rmse_x.item(),
            'rmse_y': rmse_y.item(),
            'rmse_z': rmse_z.item(),
            'rmse_total': rmse_total.item()
        }

    def compute_r2_score(self, pred: torch.Tensor, target: torch.Tensor,
                         sdf: Optional[torch.Tensor] = None, mask_inside: bool = True) -> Dict[str, float]:
        """
        计算R²决定系数

        Args:
            pred: 预测值 [batch_size, grid_points * 3]
            target: 目标值 [batch_size, grid_points * 3]
            sdf: 符号距离场 [batch_size, grid_points]
            mask_inside: 是否只计算血管内部区域

        Returns:
            r2_dict: 包含各项R²的字典
        """
        batch_size = pred.size(0)

        # 重塑
        pred_reshaped = pred.view(batch_size, -1, 3)
        target_reshaped = target.view(batch_size, -1, 3)

        # 应用掩码
        if mask_inside and sdf is not None:
            mask = (sdf > 0).float().unsqueeze(-1)
            pred_reshaped = pred_reshaped * mask
            target_reshaped = target_reshaped * mask
            total_points = mask.sum()
        else:
            total_points = batch_size * self.grid_points

        r2_scores = {}

        for i, comp in enumerate(['x', 'y', 'z']):
            pred_comp = pred_reshaped[..., i]
            target_comp = target_reshaped[..., i]

            # 计算R²
            ss_res = ((target_comp - pred_comp) ** 2).sum()
            ss_tot = ((target_comp - target_comp.mean()) ** 2).sum()

            if ss_tot > 1e-8:
                r2 = 1 - ss_res / ss_tot
            else:
                r2 = 0.0

            r2_scores[f'r2_{comp}'] = r2.item()

        # 计算总体的R²
        ss_res_total = ((target_reshaped - pred_reshaped) ** 2).sum()
        ss_tot_total = ((target_reshaped - target_reshaped.mean()) ** 2).sum()

        if ss_tot_total > 1e-8:
            r2_total = 1 - ss_res_total / ss_tot_total
        else:
            r2_total = 0.0

        r2_scores['r2_total'] = r2_total.item()

        return r2_scores

    def compute_velocity_magnitude_error(self, pred: torch.Tensor, target: torch.Tensor,
                                       sdf: Optional[torch.Tensor] = None, mask_inside: bool = True) -> Dict[str, float]:
        """
        计算速度大小误差

        Args:
            pred: 预测速度场 [batch_size, grid_points * 3]
            target: 目标速度场 [batch_size, grid_points * 3]
            sdf: 符号距离场 [batch_size, grid_points]
            mask_inside: 是否只计算血管内部区域

        Returns:
            error_dict: 包含速度大小误差的字典
        """
        batch_size = pred.size(0)

        # 重塑并计算速度大小
        pred_reshaped = pred.view(batch_size, -1, 3)
        target_reshaped = target.view(batch_size, -1, 3)

        pred_mag = torch.norm(pred_reshaped, dim=2)  # [batch_size, grid_points]
        target_mag = torch.norm(target_reshaped, dim=2)  # [batch_size, grid_points]

        # 应用掩码
        if mask_inside and sdf is not None:
            mask = (sdf > 0).float()
            pred_mag = pred_mag * mask
            target_mag = target_mag * mask
            total_points = mask.sum()
        else:
            total_points = batch_size * self.grid_points

        # 计算误差
        mag_error = torch.abs(pred_mag - target_mag)
        mae_mag = mag_error.sum() / (total_points + 1e-8)

        # 相对误差
        rel_error = mag_error / (target_mag + 1e-8)
        mae_rel = rel_error.sum() / (total_points + 1e-8)

        return {
            'mae_magnitude': mae_mag.item(),
            'mae_relative': mae_rel.item()
        }

    def compute_direction_error(self, pred: torch.Tensor, target: torch.Tensor,
                              sdf: Optional[torch.Tensor] = None, mask_inside: bool = True,
                              epsilon: float = 1e-6) -> Dict[str, float]:
        """
        计算速度方向误差（角度误差）

        Args:
            pred: 预测速度场 [batch_size, grid_points * 3]
            target: 目标速度场 [batch_size, grid_points * 3]
            sdf: 符号距离场 [batch_size, grid_points]
            mask_inside: 是否只计算血管内部区域
            epsilon: 避免除零的小值

        Returns:
            error_dict: 包含方向误差的字典
        """
        batch_size = pred.size(0)

        # 重塑
        pred_reshaped = pred.view(batch_size, -1, 3)
        target_reshaped = target.view(batch_size, -1, 3)

        # 归一化速度向量
        pred_norm = torch.norm(pred_reshaped, dim=2, keepdim=True)
        target_norm = torch.norm(target_reshaped, dim=2, keepdim=True)

        pred_unit = pred_reshaped / (pred_norm + epsilon)
        target_unit = target_reshaped / (target_norm + epsilon)

        # 计算点积（余弦相似度）
        cosine_sim = torch.sum(pred_unit * target_unit, dim=2)  # [batch_size, grid_points]
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)  # 确保在有效范围内

        # 计算角度误差（弧度）
        angle_error = torch.acos(torch.abs(cosine_sim))

        # 应用掩码
        if mask_inside and sdf is not None:
            mask = (sdf > 0).float()
            # 只对有意义的速度计算方向误差（速度大小大于阈值）
            velocity_mask = (target_norm.squeeze(-1) > epsilon).float()
            combined_mask = mask * velocity_mask
            angle_error = angle_error * combined_mask
            total_points = combined_mask.sum()
        else:
            velocity_mask = (target_norm.squeeze(-1) > epsilon).float()
            angle_error = angle_error * velocity_mask
            total_points = velocity_mask.sum()

        # 计算平均角度误差（弧度和度数）
        mean_angle_rad = angle_error.sum() / (total_points + 1e-8)
        mean_angle_deg = mean_angle_rad * 180.0 / np.pi

        return {
            'mean_angle_error_rad': mean_angle_rad.item(),
            'mean_angle_error_deg': mean_angle_deg.item()
        }

    def compute_physical_consistency(self, pred: torch.Tensor, sdf: torch.Tensor) -> Dict[str, float]:
        """
        计算物理一致性指标

        Args:
            pred: 预测速度场 [batch_size, grid_points * 3]
            sdf: 符号距离场 [batch_size, grid_points]

        Returns:
            consistency_dict: 物理一致性指标
        """
        batch_size = pred.size(0)
        D, H, W = self.grid_size

        # 重塑速度场
        pred_reshaped = pred.view(batch_size, D, H, W, 3)

        # 检查边界条件：血管外部速度应该接近零
        outside_mask = (sdf <= 0).float()
        pred_magnitude = torch.norm(pred_reshaped, dim=4)  # [batch_size, D, H, W]
        pred_magnitude_flat = pred_magnitude.view(batch_size, -1)

        outside_velocity = pred_magnitude_flat * outside_mask
        outside_violation = (outside_velocity > 0.01).float().sum()
        outside_total = outside_mask.sum()

        boundary_violation_ratio = outside_violation / (outside_total + 1e-8)

        return {
            'boundary_violation_ratio': boundary_violation_ratio.item(),
            'outside_velocity_mean': (outside_velocity.sum() / (outside_total + 1e-8)).item()
        }

    def compute_all_metrics(self, pred: torch.Tensor, target: torch.Tensor,
                           sdf: torch.Tensor) -> Dict[str, float]:
        """
        计算所有评估指标

        Args:
            pred: 预测速度场 [batch_size, grid_points * 3]
            target: 目标速度场 [batch_size, grid_points * 3]
            sdf: 符号距离场 [batch_size, grid_points]

        Returns:
            all_metrics: 包含所有指标的字典
        """
        all_metrics = {}

        # 基础误差指标
        mae_metrics = self.compute_mae(pred, target, sdf, mask_inside=True)
        rmse_metrics = self.compute_rmse(pred, target, sdf, mask_inside=True)
        r2_metrics = self.compute_r2_score(pred, target, sdf, mask_inside=True)

        # 速度特定指标
        vel_mag_metrics = self.compute_velocity_magnitude_error(pred, target, sdf, mask_inside=True)
        dir_metrics = self.compute_direction_error(pred, target, sdf, mask_inside=True)

        # 物理一致性指标
        physics_metrics = self.compute_physical_consistency(pred, sdf)

        # 合并所有指标
        all_metrics.update(mae_metrics)
        all_metrics.update(rmse_metrics)
        all_metrics.update(r2_metrics)
        all_metrics.update(vel_mag_metrics)
        all_metrics.update(dir_metrics)
        all_metrics.update(physics_metrics)

        return all_metrics


class MetricsTracker:
    """
    指标跟踪器
    用于训练过程中跟踪和记录指标
    """

    def __init__(self):
        self.metrics_history = {
            'train': [],
            'val': []
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def update(self, metrics: Dict[str, float], phase: str = 'train', epoch: int = 0):
        """
        更新指标记录

        Args:
            metrics: 指标字典
            phase: 阶段 ('train' 或 'val')
            epoch: 当前epoch
        """
        metrics_with_epoch = metrics.copy()
        metrics_with_epoch['epoch'] = epoch

        self.metrics_history[phase].append(metrics_with_epoch)

        # 更新最佳验证损失
        if phase == 'val':
            val_loss = metrics.get('total_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch

    def get_best_metrics(self, phase: str = 'val') -> Dict[str, float]:
        """
        获取最佳指标

        Args:
            phase: 阶段

        Returns:
            best_metrics: 最佳指标
        """
        if not self.metrics_history[phase]:
            return {}

        # 根据验证损失找到最佳epoch
        best_idx = min(range(len(self.metrics_history[phase])),
                      key=lambda i: self.metrics_history[phase][i].get('total_loss', float('inf')))

        return self.metrics_history[phase][best_idx]

    def get_latest_metrics(self, phase: str = 'train') -> Dict[str, float]:
        """
        获取最新的指标

        Args:
            phase: 阶段

        Returns:
            latest_metrics: 最新指标
        """
        if not self.metrics_history[phase]:
            return {}

        return self.metrics_history[phase][-1]

    def print_metrics(self, metrics: Dict[str, float], phase: str = ''):
        """
        打印指标

        Args:
            metrics: 指标字典
            phase: 阶段名称
        """
        prefix = f"[{phase.upper()}] " if phase else ""
        print(f"{prefix}Loss: {metrics.get('total_loss', 0):.6f}")

        # 打印关键指标
        if 'mae_total' in metrics:
            print(f"{prefix}MAE: {metrics['mae_total']:.6f}")
        if 'rmse_total' in metrics:
            print(f"{prefix}RMSE: {metrics['rmse_total']:.6f}")
        if 'r2_total' in metrics:
            print(f"{prefix}R²: {metrics['r2_total']:.4f}")


if __name__ == "__main__":
    # 测试评估指标
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing metrics on {device}")

    # 创建测试数据
    batch_size = 2
    grid_size = (16, 16, 16)  # 使用较小的网格进行测试
    grid_points = np.prod(grid_size)

    pred = torch.randn(batch_size, grid_points * 3).to(device)
    target = torch.randn(batch_size, grid_points * 3).to(device)
    sdf = torch.randn(batch_size, grid_points).to(device)

    print(f"Input shapes: pred={pred.shape}, target={target.shape}, sdf={sdf.shape}")

    # 创建评估器
    metrics_calculator = CFDMetrics(grid_size)

    # 计算所有指标
    all_metrics = metrics_calculator.compute_all_metrics(pred, target, sdf)

    print("\n=== 评估指标 ===")
    for key, value in all_metrics.items():
        print(f"{key}: {value:.6f}")

    # 测试指标跟踪器
    print("\n=== 测试指标跟踪器 ===")
    tracker = MetricsTracker()

    for epoch in range(3):
        train_metrics = {'total_loss': 1.0 - epoch * 0.1, 'mae_total': 0.5 - epoch * 0.05}
        val_metrics = {'total_loss': 0.9 - epoch * 0.08, 'mae_total': 0.4 - epoch * 0.04}

        tracker.update(train_metrics, 'train', epoch)
        tracker.update(val_metrics, 'val', epoch)

        tracker.print_metrics(train_metrics, 'train')
        tracker.print_metrics(val_metrics, 'val')

    print(f"\n最佳验证损失: {tracker.best_val_loss:.6f} (epoch {tracker.best_epoch})")

    print("Metrics testing completed!")