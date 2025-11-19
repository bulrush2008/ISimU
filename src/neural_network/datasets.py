"""
数据集加载和预处理模块
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
from typing import Tuple, List, Optional, Dict, Any
import warnings

# 简单的simple_train_test_split实现，避免依赖sklearn
def simple_simple_train_test_split(data_list, test_size=0.2, random_state=42):
    """简单的数据分割实现"""
    np.random.seed(random_state)
    n = len(data_list)
    indices = np.random.permutation(n)
    test_size = int(n * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    train_data = [data_list[i] for i in train_indices]
    test_data = [data_list[i] for i in test_indices]

    return train_data, test_data


class CFDataset(Dataset):
    """
    CFD数据集类
    从HDF5文件加载SDF和速度场数据
    """

    def __init__(self,
                 data_files: List[str],
                 grid_size: Tuple[int, int, int] = (64, 64, 64),
                 transform_sdf: bool = True,
                 transform_velocity: bool = True,
                 normalize: bool = True,
                 velocity_scale: float = 1.0,
                 cache_data: bool = False):
        """
        初始化数据集

        Args:
            data_files: HDF5文件路径列表
            grid_size: 网格尺寸
            transform_sdf: 是否对SDF进行变换
            transform_velocity: 是否对速度场进行变换
            normalize: 是否标准化数据
            velocity_scale: 速度场缩放因子
            cache_data: 是否缓存数据到内存
        """
        self.data_files = data_files
        self.grid_size = grid_size
        self.grid_points = np.prod(grid_size)
        self.transform_sdf = transform_sdf
        self.transform_velocity = transform_velocity
        self.normalize = normalize
        self.velocity_scale = velocity_scale
        self.cache_data = cache_data

        # 验证文件存在性
        for file_path in data_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")

        # 数据缓存
        self.data_cache = {} if cache_data else None

        # 计算数据统计信息（用于标准化）
        if normalize:
            self._compute_data_statistics()

    def _compute_data_statistics(self):
        """计算数据的统计信息用于标准化"""
        print("Computing dataset statistics...")

        all_sdf = []
        all_velocity = []

        for file_path in self.data_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    sdf = f['fields']['SDF'][:]
                    velocity = f['fields']['Velocity'][:]

                    all_sdf.append(sdf.flatten())
                    all_velocity.append(velocity.reshape(-1, 3))
            except Exception as e:
                warnings.warn(f"Failed to load {file_path}: {e}")
                continue

        if not all_sdf:
            raise ValueError("No valid data found in any file")

        # 合并所有数据
        all_sdf = np.concatenate(all_sdf, axis=0)
        all_velocity = np.concatenate(all_velocity, axis=0)

        # 计算统计信息
        self.sdf_mean = np.mean(all_sdf)
        self.sdf_std = np.std(all_sdf)

        self.velocity_mean = np.mean(all_velocity, axis=0)
        self.velocity_std = np.std(all_velocity, axis=0)

        print(f"SDF statistics: mean={self.sdf_mean:.6f}, std={self.sdf_std:.6f}")
        print(f"Velocity statistics: mean={self.velocity_mean}, std={self.velocity_std}")

    def _load_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """从HDF5文件加载数据"""
        # 检查缓存
        if self.cache_data and file_path in self.data_cache:
            return self.data_cache[file_path]

        try:
            with h5py.File(file_path, 'r') as f:
                sdf = f['fields']['SDF'][:]
                velocity = f['fields']['Velocity'][:]

                # 展平数据
                sdf_flat = sdf.flatten()
                velocity_flat = velocity.reshape(-1, 3)

                # 数据变换
                if self.transform_sdf:
                    # 对SDF进行变换，增强血管内外对比
                    sdf_flat = np.tanh(sdf_flat * 10)  # 增强对比度

                if self.transform_velocity:
                    # 对速度场进行变换
                    velocity_flat = velocity_flat * self.velocity_scale

                # 标准化
                if self.normalize:
                    sdf_flat = (sdf_flat - self.sdf_mean) / (self.sdf_std + 1e-8)
                    velocity_flat = (velocity_flat - self.velocity_mean) / (self.velocity_std + 1e-8)

                # 缓存数据
                if self.cache_data:
                    self.data_cache[file_path] = (sdf_flat, velocity_flat)

                return sdf_flat, velocity_flat

        except Exception as e:
            raise ValueError(f"Failed to load data from {file_path}: {e}")

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """获取单个数据样本"""
        file_path = self.data_files[idx]
        sdf_flat, velocity_flat = self._load_data(file_path)

        # 转换为PyTorch张量
        sdf_tensor = torch.FloatTensor(sdf_flat)
        velocity_tensor = torch.FloatTensor(velocity_flat)

        # 如果HDF5文件中包含边界条件信息，则返回
        try:
            with h5py.File(file_path, 'r') as f:
                if 'boundary_conditions' in f:
                    bc_data = f['boundary_conditions'][:]
                    bc_tensor = torch.FloatTensor(bc_data.flatten())
                    return sdf_tensor, velocity_tensor, bc_tensor
        except Exception:
            pass

        return sdf_tensor, velocity_tensor

    def get_data_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return {
            'num_samples': len(self.data_files),
            'grid_size': self.grid_size,
            'grid_points': self.grid_points,
            'input_dim': self.grid_points,
            'output_dim': self.grid_points * 3,
            'normalized': self.normalize,
            'velocity_scale': self.velocity_scale
        }


class CFDDataModule:
    """
    CFD数据模块
    管理训练、验证、测试数据集的创建和加载
    """

    def __init__(self,
                 data_dir: str,
                 grid_size: Tuple[int, int, int] = (64, 64, 64),
                 batch_size: int = 2,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 num_workers: int = 4,
                 seed: int = 42,
                 **dataset_kwargs):
        """
        初始化数据模块

        Args:
            data_dir: 数据目录路径
            grid_size: 网格尺寸
            batch_size: 批次大小
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            num_workers: 数据加载进程数
            seed: 随机种子
            **dataset_kwargs: 传递给数据集的额外参数
        """
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.dataset_kwargs = dataset_kwargs

        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        # 查找数据文件
        self.data_files = self._find_data_files()
        print(f"Found {len(self.data_files)} data files")

        # 分割数据集
        self.train_files, val_test_files = simple_train_test_split(
            self.data_files, test_size=(1 - train_ratio), random_state=seed
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        self.val_files, self.test_files = simple_train_test_split(
            val_test_files, test_size=(1 - val_size), random_state=seed
        )

        print(f"Dataset split - Train: {len(self.train_files)}, "
              f"Val: {len(self.val_files)}, Test: {len(self.test_files)}")

        # 创建数据集
        self.train_dataset = CFDataset(self.train_files, grid_size=grid_size, **dataset_kwargs)
        self.val_dataset = CFDataset(self.val_files, grid_size=grid_size, **dataset_kwargs)
        self.test_dataset = CFDataset(self.test_files, grid_size=grid_size, **dataset_kwargs)

    def _find_data_files(self) -> List[str]:
        """查找HDF5数据文件"""
        data_files = []

        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")

        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.h5'):
                file_path = os.path.join(self.data_dir, file_name)
                # 验证文件包含必要的数据
                try:
                    with h5py.File(file_path, 'r') as f:
                        if 'fields' in f and 'SDF' in f['fields'] and 'Velocity' in f['fields']:
                            data_files.append(file_path)
                except Exception:
                    continue

        if not data_files:
            raise ValueError(f"No valid HDF5 files found in {self.data_dir}")

        return sorted(data_files)

    def train_dataloader(self) -> DataLoader:
        """创建训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        """创建验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )

    def test_dataloader(self) -> DataLoader:
        """创建测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )

    def get_data_info(self) -> Dict[str, Any]:
        """获取数据模块信息"""
        return {
            'data_dir': self.data_dir,
            'total_files': len(self.data_files),
            'train_files': len(self.train_files),
            'val_files': len(self.val_files),
            'test_files': len(self.test_files),
            'grid_size': self.grid_size,
            'batch_size': self.batch_size,
            'dataset_info': self.train_dataset.get_data_info()
        }


def create_data_splits(data_files: List[str],
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    创建数据分割

    Args:
        data_files: 数据文件列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        train_files, val_files, test_files
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # 训练集和其余数据分割
    train_files, remaining_files = simple_train_test_split(
        data_files, test_size=(1 - train_ratio), random_state=seed
    )

    # 验证集和测试集分割
    val_size = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = simple_train_test_split(
        remaining_files, test_size=(1 - val_size), random_state=seed + 1
    )

    return train_files, val_files, test_files


def test_dataloader():
    """测试数据加载器"""
    # 设置数据路径
    data_dir = "matrix_data"

    if os.path.exists(data_dir):
        print("=== 测试数据加载器 ===")

        try:
            # 创建数据模块
            data_module = CFDDataModule(
                data_dir=data_dir,
                grid_size=(64, 64, 64),
                batch_size=1,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                normalize=True
            )

            # 获取数据信息
            info = data_module.get_data_info()
            print("数据信息:")
            for key, value in info.items():
                print(f"  {key}: {value}")

            # 测试数据加载
            train_loader = data_module.train_dataloader()
            print(f"\n训练数据加载器: {len(train_loader)} 批次")

            # 加载一个批次
            for batch_idx, (sdf, velocity) in enumerate(train_loader):
                print(f"批次 {batch_idx}:")
                print(f"  SDF shape: {sdf.shape}")
                print(f"  Velocity shape: {velocity.shape}")
                print(f"  SDF range: [{sdf.min():.4f}, {sdf.max():.4f}]")
                print(f"  Velocity range: [{velocity.min():.4f}, {velocity.max():.4f}]")

                if batch_idx >= 0:  # 只测试第一个批次
                    break

        except Exception as e:
            print(f"数据加载器测试失败: {e}")
    else:
        print(f"数据目录不存在: {data_dir}")


if __name__ == "__main__":
    test_dataloader()