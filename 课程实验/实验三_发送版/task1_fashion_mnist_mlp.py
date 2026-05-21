#!/usr/bin/env python
# coding: utf-8

# 任务1：Fashion-MNIST 基础多层感知机

import warnings
import copy
import random
import time
from pathlib import Path
from typing import Optional

# 画图、表格和深度学习训练需要的库
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 忽略无关紧要的警告，方便观察训练输出
warnings.filterwarnings("ignore")

# 设置绘图风格和中文显示
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 实验中会反复用到的超参数
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
HIDDEN_DIM = 256
VALID_RATIO = 0.1

# Fashion-MNIST 的 10 个类别名称
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def set_seed(seed: int) -> None:
    """固定随机种子，尽量让每次运行结果一致。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """优先使用 GPU，没有 GPU 就使用 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_plot(filename: str) -> None:
    """保存图像；在交互环境下顺便显示。"""
    output_path = Path(__file__).resolve().parent / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if "agg" in plt.get_backend().lower():
        plt.close()
    else:
        plt.show()


class FashionMLP(nn.Module):
    """一层隐藏层的基础多层感知机。"""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_classes: int = 10) -> None:
        super().__init__()
        # 先展平图片，再经过一层隐藏层和一层输出层
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_dataloaders(data_root: Path):
    """加载数据，并划分训练集、验证集、测试集。"""
    # ToTensor() 会把像素从 [0, 255] 自动缩放到 [0, 1]
    transform = transforms.ToTensor()

    # 读取完整训练集和测试集
    train_full = datasets.FashionMNIST(
        root=str(data_root),
        train=True,
        transform=transform,
        download=False,
    )
    test_dataset = datasets.FashionMNIST(
        root=str(data_root),
        train=False,
        transform=transform,
        download=False,
    )

    # 从训练集中再划出 10% 作为验证集
    valid_size = int(len(train_full) * VALID_RATIO)
    train_size = len(train_full) - valid_size
    split_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, valid_dataset = random_split(
        train_full,
        [train_size, valid_size],
        generator=split_generator,
    )

    # DataLoader 负责按 batch 读取数据
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_full, train_loader, valid_loader, test_loader


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """执行一轮训练或验证，返回平均损失和准确率。"""
    # 传入 optimizer 表示训练阶段；否则表示验证/测试阶段
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                # 训练阶段才需要反向传播和参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 统计这一批样本的损失和预测正确数量
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> pd.DataFrame:
    """训练模型，并记录每一轮的训练/验证结果。"""
    history = []
    # 保存验证集表现最好的模型参数
    best_state_dict = copy.deepcopy(model.state_dict())
    best_valid_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        valid_loss, valid_acc = run_one_epoch(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        # 如果当前验证集准确率更高，就更新最佳模型
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state_dict = copy.deepcopy(model.state_dict())

        epoch_seconds = time.time() - start_time
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_acc,
                "seconds": epoch_seconds,
            }
        )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f} | "
            f"time={epoch_seconds:.2f}s"
        )

    model.load_state_dict(best_state_dict)
    return pd.DataFrame(history)


def plot_history(history_df: pd.DataFrame) -> None:
    """绘制损失曲线和准确率曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="训练集损失")
    axes[0].plot(history_df["epoch"], history_df["valid_loss"], marker="s", label="验证集损失")
    axes[0].set_title("训练过程中的损失变化")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], marker="o", label="训练集准确率")
    axes[1].plot(history_df["epoch"], history_df["valid_accuracy"], marker="s", label="验证集准确率")
    axes[1].set_title("训练过程中的准确率变化")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()

    plt.tight_layout()
    save_plot("task1_training_curves.png")


def show_sample_images(dataset: datasets.FashionMNIST) -> None:
    """展示部分样本图片，便于直观查看类别。"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for idx, ax in enumerate(axes.flat):
        image, label = dataset[idx]
        ax.imshow(image.squeeze(0), cmap="gray")
        ax.set_title(CLASS_NAMES[label])
        ax.axis("off")

    plt.suptitle("Fashion-MNIST 样本图像")
    plt.tight_layout()
    save_plot("task1_sample_images.png")


def main() -> None:
    # 第一步：固定随机种子并确定运行设备
    set_seed(RANDOM_SEED)
    device = get_device()
    data_root = Path(__file__).resolve().parent / "fashion_mnist"

    # 第二步：加载数据，并展示一些样本图像
    train_full, train_loader, valid_loader, test_loader = build_dataloaders(data_root)
    show_sample_images(train_full)

    print("当前设备：", device)
    print("数据目录：", data_root)
    print(f"完整训练集样本数：{len(train_full)}")
    print(f"训练集批次数：{len(train_loader)}")
    print(f"验证集批次数：{len(valid_loader)}")
    print(f"测试集批次数：{len(test_loader)}")
    print("类别名称：", CLASS_NAMES)

    # 第三步：定义模型、损失函数和优化器
    # 交叉熵损失直接接收整数标签，不需要手动做 one-hot
    model = FashionMLP(hidden_dim=HIDDEN_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 第四步：开始训练，并记录每轮结果
    history_df = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    # 第五步：在测试集上做最终评估
    test_loss, test_acc = run_one_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
    )

    print("\n每轮训练记录：")
    print(history_df.to_string(index=False))

    best_row = history_df.loc[history_df["valid_accuracy"].idxmax()]
    print("\n最佳验证集结果：")
    print(
        f"epoch={int(best_row['epoch'])}, "
        f"valid_loss={best_row['valid_loss']:.4f}, "
        f"valid_accuracy={best_row['valid_accuracy']:.4f}"
    )

    print("\n测试集结果：")
    print(f"test_loss={test_loss:.4f}, test_accuracy={test_acc:.4f}")

    # 第六步：画出训练过程曲线
    plot_history(history_df)


if __name__ == "__main__":
    main()
