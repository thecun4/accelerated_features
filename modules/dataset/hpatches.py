import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import glob

class HPatchesDataset(Dataset):
    def __init__(self, root_dir, split='train', resolution=(800, 608)):
        self.root_dir = root_dir
        self.resolution = resolution
        self.sequences = []
        
        # 获取所有序列
        seq_dirs = glob.glob(os.path.join(root_dir, '*'))
        seq_dirs = [d for d in seq_dirs if os.path.isdir(d)]
        
        for seq_dir in seq_dirs:
            images = sorted(glob.glob(os.path.join(seq_dir, '*.ppm')))
            if len(images) >= 2:
                # 每个序列使用第一张图作为参考图，其他作为配对
                for i in range(1, len(images)):
                    self.sequences.append((images[0], images[i]))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.sequences[idx]
        
        # 读取图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # 转换为RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        img1 = cv2.resize(img1, self.resolution)
        img2 = cv2.resize(img2, self.resolution)
        
        # 转换为tensor并标准化
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
        
        return {
            'image0': img1,
            'image1': img2,
            'dataset_name': 'hpatches',
            'scene_id': os.path.basename(os.path.dirname(img1_path)),
            'pair_id': f"{os.path.basename(img1_path)}_{os.path.basename(img2_path)}"
        }