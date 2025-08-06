"""
Simplified XFeat training script for HPatches dataset
"""

import argparse
import os
import time
import sys

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tqdm
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# 尝试不同的导入方式
try:
    from modules.model.xfeat import XFeatModel
except ImportError:
    try:
        from modules.model import XFeatModel
    except ImportError:
        try:
            from modules.xfeat import XFeatModel
        except ImportError:
            # 如果都不行，尝试直接导入
            import modules.model as model

            XFeatModel = model.XFeatModel

from torch.utils.data import Dataset, DataLoader


def parse_arguments():
    parser = argparse.ArgumentParser(description="XFeat training script for HPatches.")

    parser.add_argument(  # HPatches数据集路径
        "--hpatches_root_path",
        type=str,
        required=True,
        help="Path to the HPatches dataset root directory.",
    )
    parser.add_argument(  # 模型保存路径
        "--ckpt_save_path",
        type=str,
        required=True,
        help="Path to save the checkpoints.",
    )
    parser.add_argument(  # 批量大小
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training. Default is 4.",
    )
    parser.add_argument(  # 训练步数
        "--n_steps",
        type=int,
        default=10000,
        help="Number of training steps. Default is 10000.",
    )
    parser.add_argument(  # 学习率
        "--lr", type=float, default=1e-4, help="Learning rate. Default is 0.0001."
    )
    parser.add_argument(  # 保存频率
        "--save_ckpt_every",
        type=int,
        default=500,
        help="Save checkpoints every N steps. Default is 500.",
    )

    return parser.parse_args()


class Trainer:
    def __init__(
        self,
        hpatches_root_path,
        ckpt_save_path,
        batch_size=4,
        n_steps=10000,
        lr=1e-4,
        save_ckpt_every=500,
    ):
        self.dev = torch.device("cpu")  # 1. 设备配置 - 强制使用CPU
        self.net = XFeatModel().to(self.dev)  # 2. 模型初始化

        # Setup optimizer
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(self.net.parameters(), lr=lr)  # 3. 优化器设置

        # 4. 加载HPatches数据集
        from modules.dataset.hpatches import HPatchesDataset

        hpatches_dataset = HPatchesDataset(hpatches_root_path, resolution=(800, 608))
        self.hpatches_loader = DataLoader(
            hpatches_dataset, batch_size=batch_size, shuffle=True
        )
        self.hpatches_iter = iter(self.hpatches_loader)

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + "/logdir", exist_ok=True)

        # 5. 日志和保存设置
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(  # TensorBoard日志
            ckpt_save_path
            + f"/logdir/xfeat_hpatches_"
            + time.strftime("%Y_%m_%d-%H_%M_%S")
        )

    def train(self):
        self.net.train()

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                try:
                    d = next(self.hpatches_iter)
                except StopIteration:
                    self.hpatches_iter = iter(self.hpatches_loader)
                    d = next(self.hpatches_iter)

                # 移动数据到设备
                for k in d.keys():
                    if isinstance(d[k], torch.Tensor):
                        d[k] = d[k].to(self.dev)

                p1, p2 = d["image0"], d["image1"]

                # RGB -> GRAY
                p1 = p1.mean(1, keepdim=True)
                p2 = p2.mean(1, keepdim=True)

                # Forward pass
                feats1, kpts1, hmap1 = self.net(p1)
                feats2, kpts2, hmap2 = self.net(p2)

                # 简化的损失计算
                loss = 0
                for b in range(p1.shape[0]):
                    f1 = feats1[b].flatten(1)  # [C, H*W]
                    f2 = feats2[b].flatten(1)  # [C, H*W]

                    # 计算特征相似性矩阵
                    sim_matrix = torch.mm(f1.t(), f2)  # [H*W, H*W]

                    # 简单的对比学习损失
                    targets = torch.arange(sim_matrix.size(0), device=self.dev)
                    loss += F.cross_entropy(sim_matrix, targets)

                loss = loss / p1.shape[0]

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()

                if (i + 1) % self.save_ckpt_every == 0:
                    print(f"Saving checkpoint at step {i + 1}")
                    torch.save(
                        self.net.state_dict(),
                        self.ckpt_save_path + f"/xfeat_hpatches_{i + 1}.pth",
                    )

                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)

                # Log metrics
                self.writer.add_scalar("Loss/total", loss.item(), i)


if __name__ == "__main__":
    args = parse_arguments()

    trainer = Trainer(
        hpatches_root_path=args.hpatches_root_path,
        ckpt_save_path=args.ckpt_save_path,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        save_ckpt_every=args.save_ckpt_every,
    )

    trainer.train()
