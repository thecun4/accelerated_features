import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleXFeat(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        
        # 简化的特征提取网络
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, feature_dim, 3, padding=1),
            nn.ReLU(),
        )
        
        # 特征描述子头
        self.descriptor_head = nn.Conv2d(feature_dim, feature_dim, 1)
        
    def forward(self, x):
        features = self.backbone(x)
        descriptors = self.descriptor_head(features)
        descriptors = F.normalize(descriptors, p=2, dim=1)
        return descriptors

def create_model():
    """创建轻量级XFeat模型"""
    return SimpleXFeat(feature_dim=64)