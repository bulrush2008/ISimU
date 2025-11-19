"""
训练器和训练循环实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
# from tqdm import tqdm  # 暂时注释掉避免依赖问题
import warnings

from .models import create_model
from .losses import create_loss_function
from .metrics import CFDMetrics, MetricsTracker


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 50, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        检查是否应该早停

        Args:
            val_loss: 验证损失
            model: 模型

        Returns:
            should_stop: 是否应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True

        return False


class LearningRateScheduler:
    """学习率调度器管理"""

    def __init__(self, optimizer: optim.Optimizer, scheduler_type: str = 'cosine', **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        if scheduler_type.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=kwargs.get('T_max', 100), eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type.lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=kwargs.get('step_size', 30), gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type.lower() == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=kwargs.get('gamma', 0.95)
            )
        elif scheduler_type.lower() == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=kwargs.get('factor', 0.5), patience=kwargs.get('patience', 10)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def step(self, metric: Optional[float] = None):
        """更新学习率"""
        if self.scheduler_type.lower() == 'reduce_on_plateau' and metric is not None:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        return self.scheduler.get_lr()


class ModelTrainer:
    """
    模型训练器
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化训练器

        Args:
            model: 神经网络模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            config: 训练配置
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 默认配置
        self.config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'epochs': 1000,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'optimizer': 'Adam',
            'loss_function': 'combined',
            'scheduler': 'cosine',
            'early_stopping_patience': 50,
            'gradient_clip_value': 1.0,
            'save_best_model': True,
            'save_interval': 10,
            'log_interval': 1,
            'eval_interval': 1
        }

        # 更新配置
        if config:
            self.config.update(config)

        # 设置设备
        self.device = torch.device(self.config['device'])
        self.model.to(self.device)

        # 获取网格尺寸（从数据集中推断）
        sample_batch = next(iter(train_loader))
        self.grid_size = self._infer_grid_size(sample_batch[0].shape[1])

        # 创建优化器
        self.optimizer = self._create_optimizer()

        # 创建损失函数
        self.loss_function = create_loss_function(
            self.config['loss_function'],
            **self.config.get('loss_kwargs', {})
        )

        # 创建学习率调度器
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            self.config['scheduler'],
            **self.config.get('scheduler_kwargs', {})
        )

        # 创建早停机制
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience']
        )

        # 创建评估器
        self.metrics_calculator = CFDMetrics(self.grid_size)

        # 创建指标跟踪器
        self.metrics_tracker = MetricsTracker()

        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []

    def _infer_grid_size(self, input_dim: int) -> Tuple[int, int, int]:
        """从输入维度推断网格尺寸"""
        # 假设是立方体网格
        grid_size = int(round(input_dim ** (1/3)))
        return (grid_size, grid_size, grid_size)

    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_name = self.config['optimizer'].lower()
        lr = self.config['learning_rate']
        weight_decay = self.config['weight_decay']

        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = []

        # progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config['epochs']}")
        # 简单的进度显示替代tqdm
        total_batches = len(self.train_loader)
        print(f"开始 Epoch {self.current_epoch + 1}/{self.config['epochs']}, 总批次: {total_batches}")

        for batch_idx, batch_data in enumerate(self.train_loader):
            # 处理可能包含边界条件的数据批次
            if len(batch_data) == 3:
                sdf, velocity, bc_params = batch_data
                sdf = sdf.to(self.device)
                velocity = velocity.to(self.device)
                bc_params = bc_params.to(self.device)
                has_bc = True
            else:
                sdf, velocity = batch_data
                sdf = sdf.to(self.device)
                velocity = velocity.to(self.device)
                bc_params = None
                has_bc = False

            # 前向传播
            self.optimizer.zero_grad()
            pred_velocity = self.model(sdf, bc_params)

            # 计算损失
            if isinstance(self.loss_function, dict):
                # 物理信息损失函数返回字典
                loss_dict = self.loss_function(pred_velocity, velocity, sdf, self.grid_size)
                loss = loss_dict['total_loss']
            else:
                # 标准损失函数
                loss = self.loss_function(pred_velocity, velocity, sdf)
                loss_dict = {'total_loss': loss.item()}

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config['gradient_clip_value'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip_value']
                )

            self.optimizer.step()

            # 记录损失
            epoch_losses.append(loss.item())

            # 计算指标（每log_interval批次计算一次）
            if batch_idx % self.config['log_interval'] == 0:
                with torch.no_grad():
                    metrics = self.metrics_calculator.compute_all_metrics(
                        pred_velocity, velocity, sdf
                    )
                    metrics.update(loss_dict)
                    epoch_metrics.append(metrics)

                # 简单的进度显示
                if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                    print(f"  批次 {batch_idx+1}/{total_batches} - "
                          f"Loss: {loss.item():.6f}, "
                          f"MAE: {metrics.get('mae_total', 0):.6f}, "
                          f"RMSE: {metrics.get('rmse_total', 0):.6f}")

        # 计算epoch平均指标
        avg_loss = np.mean(epoch_losses)
        if epoch_metrics:
            avg_metrics = {}
            for key in epoch_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        else:
            avg_metrics = {'total_loss': avg_loss}

        return avg_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        val_losses = []
        val_metrics = []

        with torch.no_grad():
            for batch_data in self.val_loader:
                # 处理可能包含边界条件的数据批次
                if len(batch_data) == 3:
                    sdf, velocity, bc_params = batch_data
                    sdf = sdf.to(self.device)
                    velocity = velocity.to(self.device)
                    bc_params = bc_params.to(self.device)
                else:
                    sdf, velocity = batch_data
                    sdf = sdf.to(self.device)
                    velocity = velocity.to(self.device)
                    bc_params = None

                # 前向传播
                pred_velocity = self.model(sdf, bc_params)

                # 计算损失
                if isinstance(self.loss_function, dict):
                    loss_dict = self.loss_function(pred_velocity, velocity, sdf, self.grid_size)
                    loss = loss_dict['total_loss']
                else:
                    loss = self.loss_function(pred_velocity, velocity, sdf)
                    loss_dict = {'total_loss': loss.item()}

                val_losses.append(loss.item())

                # 计算指标
                metrics = self.metrics_calculator.compute_all_metrics(
                    pred_velocity, velocity, sdf
                )
                metrics.update(loss_dict)
                val_metrics.append(metrics)

        # 计算平均指标
        avg_loss = np.mean(val_losses)
        avg_metrics = {}
        for key in val_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in val_metrics])

        return avg_metrics

    def train(self, save_dir: str = 'models') -> Dict[str, Any]:
        """
        完整训练流程

        Args:
            save_dir: 模型保存目录

        Returns:
            training_results: 训练结果
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        print(f"开始训练，设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"网格尺寸: {self.grid_size}")
        print(f"训练数据: {len(self.train_loader.dataset)} 样本")
        print(f"验证数据: {len(self.val_loader.dataset)} 样本")

        start_time = time.time()

        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch

            # 训练阶段
            train_metrics = self.train_epoch()

            # 验证阶段
            if epoch % self.config['eval_interval'] == 0:
                val_metrics = self.validate_epoch()

                # 更新学习率
                self.scheduler.step(val_metrics['total_loss'])

                # 记录指标
                self.metrics_tracker.update(train_metrics, 'train', epoch)
                self.metrics_tracker.update(val_metrics, 'val', epoch)

                # 打印epoch结果
                print(f"\nEpoch {epoch + 1}/{self.config['epochs']}:")
                print(f"  训练 - Loss: {train_metrics['total_loss']:.6f}, "
                      f"MAE: {train_metrics.get('mae_total', 0):.6f}")
                print(f"  验证 - Loss: {val_metrics['total_loss']:.6f}, "
                      f"MAE: {val_metrics.get('mae_total', 0):.6f}, "
                      f"R²: {val_metrics.get('r2_total', 0):.4f}")
                print(f"  学习率: {self.scheduler.get_lr()[0]:.2e}")

                # 保存最佳模型
                if self.config['save_best_model']:
                    current_val_loss = val_metrics['total_loss']
                    if current_val_loss < self.best_val_loss:
                        self.best_val_loss = current_val_loss
                        self.save_model(save_dir, epoch, is_best=True)
                        print(f"  保存最佳模型 (验证损失: {current_val_loss:.6f})")

                # 定期保存模型
                if epoch % self.config['save_interval'] == 0:
                    self.save_model(save_dir, epoch)

                # 早停检查
                if self.early_stopping(val_metrics['total_loss'], self.model):
                    print(f"\n早停触发，在epoch {epoch + 1}停止训练")
                    break

        # 训练完成
        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")

        # 最终评估
        if self.test_loader:
            print("正在进行最终测试...")
            test_results = self.evaluate(self.test_loader)
            print(f"测试集结果 - Loss: {test_results['total_loss']:.6f}, "
                  f"MAE: {test_results.get('mae_total', 0):.6f}, "
                  f"R²: {test_results.get('r2_total', 0):.4f}")

        # 保存最终模型
        self.save_model(save_dir, self.current_epoch, is_final=True)

        # 保存训练历史
        self.save_training_history(save_dir)

        return {
            'total_epochs': self.current_epoch + 1,
            'best_val_loss': self.best_val_loss,
            'training_time': total_time,
            'final_metrics': self.metrics_tracker.get_latest_metrics('val')
        }

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        all_metrics = []

        with torch.no_grad():
            for batch_data in data_loader:
                # 处理可能包含边界条件的数据批次
                if len(batch_data) == 3:
                    sdf, velocity, bc_params = batch_data
                    sdf = sdf.to(self.device)
                    velocity = velocity.to(self.device)
                    bc_params = bc_params.to(self.device)
                else:
                    sdf, velocity = batch_data
                    sdf = sdf.to(self.device)
                    velocity = velocity.to(self.device)
                    bc_params = None

                # 前向传播
                pred_velocity = self.model(sdf, bc_params)

                # 计算指标
                metrics = self.metrics_calculator.compute_all_metrics(
                    pred_velocity, velocity, sdf
                )

                # 计算损失
                if isinstance(self.loss_function, dict):
                    loss_dict = self.loss_function(pred_velocity, velocity, sdf, self.grid_size)
                    metrics.update(loss_dict)
                else:
                    loss = self.loss_function(pred_velocity, velocity, sdf)
                    metrics['total_loss'] = loss.item()

                all_metrics.append(metrics)

        # 计算平均指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics

    def save_model(self, save_dir: str, epoch: int, is_best: bool = False, is_final: bool = False):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'grid_size': self.grid_size
        }

        if is_best:
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
        elif is_final:
            torch.save(checkpoint, os.path.join(save_dir, 'final_model.pth'))
        else:
            torch.save(checkpoint, os.path.join(save_dir, f'model_epoch_{epoch}.pth'))

    def load_model(self, checkpoint_path: str):
        """加载模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"模型加载成功，epoch: {self.current_epoch}, 最佳验证损失: {self.best_val_loss:.6f}")

    def save_training_history(self, save_dir: str):
        """保存训练历史"""
        history = {
            'config': self.config,
            'grid_size': self.grid_size,
            'train_history': self.metrics_tracker.metrics_history['train'],
            'val_history': self.metrics_tracker.metrics_history['val'],
            'best_epoch': self.metrics_tracker.best_epoch,
            'best_val_loss': self.metrics_tracker.best_val_loss
        }

        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        print(f"训练历史已保存到: {os.path.join(save_dir, 'training_history.json')}")


def create_trainer(model_type: str = 'fc',
                  data_module: Any = None,
                  config: Optional[Dict[str, Any]] = None) -> ModelTrainer:
    """
    创建训练器的工厂函数

    Args:
        model_type: 模型类型
        data_module: 数据模块
        config: 训练配置

    Returns:
        trainer: 训练器
    """
    if data_module is None:
        raise ValueError("data_module is required")

    # 获取数据信息
    data_info = data_module.get_data_info()
    grid_size = data_info['grid_size']

    # 创建模型
    model = create_model(
        model_type=model_type,
        grid_size=grid_size,
        **config.get('model_kwargs', {})
    )

    # 创建训练器
    trainer = ModelTrainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        test_loader=data_module.test_dataloader(),
        config=config
    )

    return trainer


if __name__ == "__main__":
    # 测试训练器
    print("=== 测试训练器 ===")
    print("注意: 需要先有数据文件才能完整测试训练器")

    # 这里只测试训练器的基本初始化
    from .datasets import CFDDataModule

    # 如果有数据文件，可以取消注释以下代码进行测试
    """
    try:
        # 创建数据模块
        data_module = CFDDataModule(
            data_dir="matrix_data",
            grid_size=(64, 64, 64),
            batch_size=1,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )

        # 创建训练配置
        config = {
            'epochs': 5,  # 短时间测试
            'learning_rate': 1e-4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        # 创建训练器
        trainer = create_trainer(
            model_type='fc',
            data_module=data_module,
            config=config
        )

        print("训练器创建成功！")
        print(f"模型参数数量: {sum(p.numel() for p in trainer.model.parameters()):,}")

    except Exception as e:
        print(f"训练器测试失败: {e}")
    """

    print("训练器模块测试完成！")