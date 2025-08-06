"""
"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import argparse
import os
import time
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="XFeat training script.")

    # 数据集路径
    parser.add_argument(
        "--megadepth_root_path",
        type=str,
        default="/ssd/guipotje/Data/MegaDepth",
        help="Path to the MegaDepth dataset root directory.",
    )
    parser.add_argument(
        "--synthetic_root_path",
        type=str,
        default="/homeLocal/guipotje/sshfs/datasets/coco_20k",
        help="Path to the synthetic dataset root directory.",
    )

    # 训练配置
    parser.add_argument(
        "--ckpt_save_path",
        type=str,
        required=True,
        help="Path to save the checkpoints.",
    )  # 检查点存放路径
    parser.add_argument(  # xfeat_default: MegaDepth + 合成数据；xfeat_synthetic: 仅合成数据；xfeat_megadepth: 仅MegaDepth数据
        "--training_type",
        type=str,
        default="xfeat_default",
        choices=["xfeat_default", "xfeat_synthetic", "xfeat_megadepth"],
        help="Training scheme. xfeat_default uses both megadepth & synthetic warps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for training. Default is 10.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=160_000,  # 16万步训练
        help="Number of training steps. Default is 160000.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,  # 学习率
        help="Learning rate. Default is 0.0003.",
    )
    parser.add_argument(
        "--gamma_steplr",
        type=float,
        default=0.5,
        help="Gamma value for StepLR scheduler. Default is 0.5.",
    )
    parser.add_argument(
        "--training_res",
        type=lambda s: tuple(map(int, s.split(","))),
        default=(800, 608),
        help="Training resolution as width,height. Default is (800, 608).",
    )
    parser.add_argument(
        "--device_num",
        type=str,
        default="0",
        help='Device number to use for training. Default is "0".',
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, perform a dry run training with a mini-batch for sanity check.",
    )
    parser.add_argument(
        "--save_ckpt_every",
        type=int,
        default=500,
        help="Save checkpoints every N steps. Default is 500.",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num

    return args


args = parse_arguments()

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from modules.model import *
from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *

from modules.dataset.megadepth.megadepth import MegaDepthDataset
from modules.dataset.megadepth import megadepth_warper
from torch.utils.data import Dataset, DataLoader


class Trainer:
    """
    Class for training XFeat with default params as described in the paper.
    We use a blend of MegaDepth (labeled) pairs with synthetically warped images (self-supervised).
    The major bottleneck is to keep loading huge megadepth h5 files from disk,
    the network training itself is quite fast.
    """

    def __init__(
        self,
        megadepth_root_path,
        synthetic_root_path,
        ckpt_save_path,
        model_name="xfeat_default",
        batch_size=10,
        n_steps=160_000,
        lr=3e-4,
        gamma_steplr=0.5,
        training_res=(800, 608),
        device_num="0",
        dry_run=False,
        save_ckpt_every=500,
    ):
        # 设备和模型初始化
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = XFeatModel().to(self.dev)

        # Setup optimizer # 优化器设置
        self.batch_size = batch_size # 每次训练迭代中处理的样本数量
        self.steps = n_steps
        self.opt = optim.Adam(
            filter(lambda x: x.requires_grad, self.net.parameters()), lr=lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=30_000, gamma=gamma_steplr
        )

        ##################### Synthetic COCO INIT ##########################
        if model_name in ("xfeat_default", "xfeat_synthetic"):
            self.augmentor = AugmentationPipe(
                img_dir=synthetic_root_path,
                device=self.dev,
                load_dataset=True,
                batch_size=int(
                    self.batch_size * 0.4  # 40%批量用于合成数据
                    if model_name == "xfeat_default"
                    else batch_size
                ),
                out_resolution=training_res,
                warp_resolution=training_res,
                sides_crop=0.1,
                max_num_imgs=3_000,
                num_test_imgs=5,
                photometric=True,  # 光度变换
                geometric=True,  # 几何变换
                reload_step=4_000,  # 每4000步重新加载数据
            )
        else:
            self.augmentor = None
        ##################### Synthetic COCO END #######################

        ##################### MEGADEPTH INIT ##########################
        if model_name in ("xfeat_default", "xfeat_megadepth"):
            TRAIN_BASE_PATH = f"{megadepth_root_path}/train_data/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"

            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

            # 加载所有NPZ元数据文件
            npz_paths = glob.glob(TRAIN_NPZ_ROOT + "/*.npz")[:]
            data = torch.utils.data.ConcatDataset(
                [
                    MegaDepthDataset(root_dir=TRAINVAL_DATA_SOURCE, npz_path=path)
                    for path in tqdm.tqdm(
                        npz_paths, desc="[MegaDepth] Loading metadata"
                    )
                ]
            )

            self.data_loader = DataLoader(
                data,
                batch_size=int(
                    self.batch_size * 0.6  # 60%批量用于MegaDepth
                    if model_name == "xfeat_default"
                    else batch_size
                ),
                shuffle=True,
            )
            self.data_iter = iter(self.data_loader)

        else:
            self.data_iter = None
        ##################### MEGADEPTH INIT END #######################

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + "/logdir", exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(
            ckpt_save_path
            + f"/logdir/{model_name}_"
            + time.strftime("%Y_%m_%d-%H_%M_%S")
        )
        self.model_name = model_name

    def train(self):
        self.net.train()

        difficulty = 0.10

        p1s, p2s, H1, H2 = None, None, None, None
        d = None

        # 数据准备阶段
        if self.augmentor is not None:  # 获取合成数据
            p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

        if self.data_iter is not None:  # 获取MegaDepth数据
            d = next(self.data_iter)

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                if not self.dry_run:
                    if self.data_iter is not None:
                        try:
                            # Get the next MD batch
                            d = next(self.data_iter)

                        except StopIteration:
                            print("End of DATASET!")
                            # If StopIteration is raised, create a new iterator.
                            self.data_iter = iter(self.data_loader)  # 重新开始迭代
                            d = next(self.data_iter)

                    if self.augmentor is not None:
                        # Grab synthetic data
                        p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

                # 对应点生成
                if d is not None:  # MegaDepth对应点（基于深度图）
                    for k in d.keys():
                        if isinstance(d[k], torch.Tensor):
                            d[k] = d[k].to(self.dev)

                    p1, p2 = d["image0"], d["image1"]
                    positives_md_coarse = megadepth_warper.spvs_coarse(d, 8)

                if self.augmentor is not None:  # 合成数据对应点（基于单应性变换）
                    h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                    _, positives_s_coarse = get_corresponding_pts(
                        p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse
                    )

                # Join megadepth & synthetic data
                with torch.inference_mode():
                    # RGB -> GRAY
                    if d is not None:  # RGB转灰度
                        p1 = p1.mean(1, keepdim=True)
                        p2 = p2.mean(1, keepdim=True)
                    if self.augmentor is not None:
                        p1s = p1s.mean(1, keepdim=True)
                        p2s = p2s.mean(1, keepdim=True)

                    # Cat two batches # 根据训练模式合并数据
                    if self.model_name in ("xfeat_default"):
                        p1 = torch.cat([p1s, p1], dim=0)  # 合并合成+真实数据
                        p2 = torch.cat([p2s, p2], dim=0)
                        positives_c = positives_s_coarse + positives_md_coarse
                    elif self.model_name in ("xfeat_synthetic"):
                        p1 = p1s
                        p2 = p2s  # 仅合成数据
                        positives_c = positives_s_coarse
                    else:
                        positives_c = positives_md_coarse  # 仅MegaDepth数据

                # 前向传播
                # Check if batch is corrupted with too few correspondences # 质量检查：过滤对应点过少的样本
                is_corrupted = False
                for p in positives_c:
                    if len(p) < 30:
                        is_corrupted = True

                if is_corrupted:
                    continue

                # Forward pass # 网络前向传播
                feats1, kpts1, hmap1 = self.net(p1)  # 特征、关键点、热图
                feats2, kpts2, hmap2 = self.net(p2)

                # 损失计算
                loss_items = []

                for b in range(len(positives_c)):
                    # Get positive correspondencies # 获取对应点坐标
                    pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                    # Grab features at corresponding idxs # 提取对应位置的特征
                    m1 = feats1[b, :, pts1[:, 1].long(), pts1[:, 0].long()].permute(
                        1, 0
                    )
                    m2 = feats2[b, :, pts2[:, 1].long(), pts2[:, 0].long()].permute(
                        1, 0
                    )

                    # grab heatmaps at corresponding idxs # 提取对应位置的关键点热图
                    h1 = hmap1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()]
                    h2 = hmap2[b, 0, pts2[:, 1].long(), pts2[:, 0].long()]
                    coords1 = self.net.fine_matcher(
                        torch.cat([m1, m2], dim=-1)
                    )  # 精细匹配

                    # Compute losses # 计算各种损失
                    loss_ds, conf = dual_softmax_loss(m1, m2)  # 特征匹配损失
                    loss_coords, acc_coords = (
                        coordinate_classification_loss(  # 坐标回归损失
                            coords1, pts1, pts2, conf
                        )
                    )
                    # 关键点定位损失（使用ALIKE蒸馏）
                    loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b])
                    loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b])
                    loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2) * 2.0
                    acc_pos = (acc_pos1 + acc_pos2) / 2

                    loss_kp = keypoint_loss(h1, conf) + keypoint_loss(
                        h2, conf
                    )  # 关键点可靠性损失

                    # 收集所有损失
                    loss_items.append(loss_ds.unsqueeze(0))
                    loss_items.append(loss_coords.unsqueeze(0))
                    loss_items.append(loss_kp.unsqueeze(0))
                    loss_items.append(loss_kp_pos.unsqueeze(0))

                    if b == 0:
                        acc_coarse_0 = check_accuracy(m1, m2)

                acc_coarse = check_accuracy(m1, m2)

                # 反向传播和优化
                nb_coarse = len(m1)
                loss = torch.cat(loss_items, -1).mean()  # 计算总损失
                loss_coarse = loss_ds.item()
                loss_coord = loss_coords.item()
                loss_coord = loss_coords.item()
                loss_kp_pos = loss_kp_pos.item()
                loss_l1 = loss_kp.item()

                # Compute Backward Pass
                loss.backward()  # 反向传播
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)  # 梯度裁剪
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()  # 学习率调度

                if (i + 1) % self.save_ckpt_every == 0:  # 定期保存模型
                    print("saving iter ", i + 1)
                    torch.save(
                        self.net.state_dict(),
                        self.ckpt_save_path + f"/{self.model_name}_{i + 1}.pth",
                    )

                pbar.set_description(
                    "Loss: {:.4f} acc_c0 {:.3f} acc_c1 {:.3f} acc_f: {:.3f} loss_c: {:.3f} loss_f: {:.3f} loss_kp: {:.3f} #matches_c: {:d} loss_kp_pos: {:.3f} acc_kp_pos: {:.3f}".format(
                        loss.item(),
                        acc_coarse_0,
                        acc_coarse,
                        acc_coords,
                        loss_coarse,
                        loss_coord,
                        loss_l1,
                        nb_coarse,
                        loss_kp_pos,
                        acc_pos,
                    )
                )
                pbar.update(1)

                # Log metrics # TensorBoard日志记录
                self.writer.add_scalar("Loss/total", loss.item(), i)
                self.writer.add_scalar("Accuracy/coarse_synth", acc_coarse_0, i)
                self.writer.add_scalar("Accuracy/coarse_mdepth", acc_coarse, i)
                self.writer.add_scalar("Accuracy/fine_mdepth", acc_coords, i)
                self.writer.add_scalar("Accuracy/kp_position", acc_pos, i)
                self.writer.add_scalar("Loss/coarse", loss_coarse, i)
                self.writer.add_scalar("Loss/fine", loss_coord, i)
                self.writer.add_scalar("Loss/reliability", loss_l1, i)
                self.writer.add_scalar("Loss/keypoint_pos", loss_kp_pos, i)
                self.writer.add_scalar("Count/matches_coarse", nb_coarse, i)


if __name__ == "__main__":
    trainer = Trainer(
        megadepth_root_path=args.megadepth_root_path,
        synthetic_root_path=args.synthetic_root_path,
        ckpt_save_path=args.ckpt_save_path,
        model_name=args.training_type,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every,
    )

    # The most fun part
    trainer.train()
