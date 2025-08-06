import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import random


class HpatchesDataset(Dataset):
    def __init__(self, hpatches_dir, image_size=256, split="train", train_ratio=0.8):
        self.hpatches_dir = Path(hpatches_dir)
        self.image_size = image_size
        self.split = split
        self.sequences = list(self.hpatches_dir.glob("*"))
        self.image_pairs = self._collect_image_pairs()
        self._split_data(train_ratio)

    def _collect_image_pairs(self):
        pairs = []
        for seq_dir in self.sequences:
            if seq_dir.is_dir():
                images = sorted(list(seq_dir.glob("*.ppm")))
                # 创建图像对（参考图像和变换图像）
                ref_img = images[0]  # 第一张作为参考
                for img in images[1:]:
                    pairs.append((ref_img, img))
        return pairs

    def _split_data(self, train_ratio):
        """划分训练集和测试集"""
        random.seed(42)  # 设置随机种子确保可重复
        random.shuffle(self.image_pairs)

        total_pairs = len(self.image_pairs)
        train_size = int(total_pairs * train_ratio)

        if self.split == "train":
            self.image_pairs = self.image_pairs[:train_size]
            # 限制训练数据量以适应CPU训练
            self.image_pairs = self.image_pairs[:1000]
        elif self.split == "test":
            self.image_pairs = self.image_pairs[train_size:]
            # 限制测试数据量
            self.image_pairs = self.image_pairs[:200]

        print(f"{self.split.capitalize()} set size: {len(self.image_pairs)}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]

        # 读取图像
        img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)

        # 检查图像是否成功读取
        if img1 is None or img2 is None:
            print(f"Warning: Failed to load images {img1_path} or {img2_path}")
            # 返回零图像作为fallback
            img1 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            img2 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # 调整大小
        img1 = cv2.resize(img1, (self.image_size, self.image_size))
        img2 = cv2.resize(img2, (self.image_size, self.image_size))

        # 转换为张量
        img1 = torch.from_numpy(img1).float() / 255.0
        img2 = torch.from_numpy(img2).float() / 255.0

        img1 = img1.unsqueeze(0)  # 添加通道维度
        img2 = img2.unsqueeze(0)

        return img1, img2


def create_dataloader(
    hpatches_dir, batch_size=2, image_size=256, split="train", train_ratio=0.8
):
    """创建数据加载器，支持训练集和测试集"""
    dataset = HpatchesDataset(hpatches_dir, image_size, split, train_ratio)

    # 测试集不需要shuffle
    shuffle = True if split == "train" else False

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def create_train_test_loaders(
    hpatches_dir, batch_size=2, image_size=256, train_ratio=0.8
):
    """同时创建训练和测试数据加载器"""
    train_loader = create_dataloader(
        hpatches_dir, batch_size, image_size, "train", train_ratio
    )
    test_loader = create_dataloader(
        hpatches_dir, batch_size, image_size, "test", train_ratio
    )

    return train_loader, test_loader
