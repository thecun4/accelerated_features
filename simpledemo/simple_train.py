import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import os

from simple_xfeat import create_model
from data_loader import create_train_test_loaders


def improved_contrastive_loss(desc1, desc2, margin=1.0):
    """改进的对比损失函数"""
    batch_size = desc1.size(0)

    # 展平特征描述子
    desc1_flat = desc1.view(batch_size, desc1.size(1), -1)
    desc2_flat = desc2.view(batch_size, desc2.size(1), -1)

    # 计算每个位置的相似度
    desc1_norm = F.normalize(desc1_flat, p=2, dim=1)
    desc2_norm = F.normalize(desc2_flat, p=2, dim=1)

    # 计算正样本损失 (同一图像对的对应位置应该相似)
    positive_similarity = torch.sum(desc1_norm * desc2_norm, dim=1)
    positive_loss = torch.mean(1 - positive_similarity)

    # 计算负样本损失 (不同图像对之间应该不相似)
    negative_loss = 0
    if batch_size > 1:
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    neg_similarity = torch.sum(desc1_norm[i] * desc2_norm[j], dim=0)
                    negative_loss += torch.mean(F.relu(neg_similarity - margin + 1))

        negative_loss = negative_loss / (batch_size * (batch_size - 1))

    total_loss = positive_loss + negative_loss
    return total_loss


def simple_mse_loss(desc1, desc2):
    """简单的MSE损失作为备选"""
    # 假设对应的图像对应该有相似的特征
    return F.mse_loss(desc1, desc2)


def correlation_loss(desc1, desc2):
    """基于相关性的损失函数"""
    batch_size = desc1.size(0)

    # 展平特征
    desc1_flat = desc1.view(batch_size, -1)
    desc2_flat = desc2.view(batch_size, -1)

    # 归一化
    desc1_norm = F.normalize(desc1_flat, p=2, dim=1)
    desc2_norm = F.normalize(desc2_flat, p=2, dim=1)

    # 计算余弦相似度
    cosine_sim = torch.sum(desc1_norm * desc2_norm, dim=1)

    # 我们希望相似度接近1（完全相似）
    loss = torch.mean(1 - cosine_sim)

    return loss


def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for img1, img2 in test_loader:
            img1, img2 = img1.to(device), img2.to(device)

            desc1 = model(img1)
            desc2 = model(img2)

            # 使用改进的损失函数
            loss = correlation_loss(desc1, desc2)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def train_with_validation():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 指定你的hpatches数据集路径
    hpatches_dir = "/home/mol/work/hpatches-sequences-release"

    # 创建训练和测试数据加载器
    train_loader, test_loader = create_train_test_loaders(
        hpatches_dir,
        batch_size=4,  # 稍微增加batch size以便计算负样本
        image_size=128,
        train_ratio=0.8,
    )

    # 创建模型
    model = create_model().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)  # 稍微提高学习率

    # 训练循环
    num_epochs = 20
    best_test_loss = float("inf")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        for batch_idx, (img1, img2) in enumerate(pbar):
            img1, img2 = img1.to(device), img2.to(device)

            desc1 = model(img1)
            desc2 = model(img2)

            # 使用改进的损失函数
            loss = correlation_loss(desc1, desc2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_train_batches += 1

            pbar.set_postfix({"Train Loss": f"{loss.item():.4f}"})

            # CPU训练时限制批次
            if batch_idx >= 100:  # 增加训练批次
                break

        avg_train_loss = train_loss / num_train_batches

        # 测试阶段
        test_loss = evaluate_model(model, test_loader, device)

        print(f"Epoch {epoch + 1}:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  测试损失: {test_loss:.4f}")

        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_xfeat_model.pth")
            print(f"  保存最佳模型 (测试损失: {test_loss:.4f})")

        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "test_loss": test_loss,
                },
                f"checkpoint_epoch_{epoch + 1}.pth",
            )

        print()

    print("训练完成！")
    print(f"最佳测试损失: {best_test_loss:.4f}")


if __name__ == "__main__":
    train_with_validation()
