"""
神经网络模块
包含CFD代理模型的架构、训练、评估等功能
"""

from .models import (
    SDFToVelocityNet,
    ConvolutionalSDFNet,
    HybridSDFNet,
    create_model
)

from .datasets import (
    CFDataset,
    CFDDataModule,
    create_data_splits
)

from .losses import (
    MSELossWithMask,
    WeightedMSELoss,
    PhysicsInformedLoss,
    CombinedLoss,
    create_loss_function
)

from .metrics import (
    CFDMetrics,
    MetricsTracker
)

from .trainers import (
    ModelTrainer,
    EarlyStopping,
    LearningRateScheduler,
    create_trainer
)

__all__ = [
    # Models
    'SDFToVelocityNet',
    'ConvolutionalSDFNet',
    'HybridSDFNet',
    'create_model',

    # Datasets
    'CFDataset',
    'CFDDataModule',
    'create_data_splits',

    # Losses
    'MSELossWithMask',
    'WeightedMSELoss',
    'PhysicsInformedLoss',
    'CombinedLoss',
    'create_loss_function',

    # Metrics
    'CFDMetrics',
    'MetricsTracker',

    # Trainers
    'ModelTrainer',
    'EarlyStopping',
    'LearningRateScheduler',
    'create_trainer'
]