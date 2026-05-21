#!/usr/bin/env python
# coding: utf-8

# 任务2：比较不同激活函数的影响

import warnings
import copy
import random
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 忽略无关紧要的警告，方便看训练输出
warnings.filterwarnings("ignore")

# 设置绘图风格和中文显示
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 本任务固定不变的超参数
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
HIDDEN_DIM = 256
VALID_RATIO = 0.1

# 任务要求对比的三种激活函数
ACTIVATION_FACTORIES = {
    "ReLU": nn.ReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
}


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


class ActivationMLP(nn.Module):
    """一层隐藏层的 MLP，只更换隐藏层激活函数。"""

    def __init__(self, activation_cls, hidden_dim: int = HIDDEN_DIM, num_classes: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            activation_cls(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_dataloaders(data_root: Path):
    """加载数据，并划分训练集、验证集、测试集。"""
    transform = transforms.ToTensor()

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

    valid_size = int(len(train_full) * VALID_RATIO)
    train_size = len(train_full) - valid_size
    split_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, valid_dataset = random_split(
        train_full,
        [train_size, valid_size],
        generator=split_generator,
    )

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, valid_loader, test_loader


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """执行一轮训练或验证，返回平均损失和准确率。"""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def train_model(
    activation_name: str,
    activation_cls,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame]:
    """训练某一种激活函数对应的模型，并记录每轮结果。"""
    set_seed(RANDOM_SEED)
    model = ActivationMLP(activation_cls=activation_cls).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_state_dict = copy.deepcopy(model.state_dict())
    best_valid_acc = 0.0

    print(f"\n========== 开始训练：{activation_name} ==========")

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

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state_dict = copy.deepcopy(model.state_dict())

        epoch_seconds = time.time() - start_time
        history.append(
            {
                "activation": activation_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_acc,
                "seconds": epoch_seconds,
            }
        )

        print(
            f"{activation_name} | Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f} | "
            f"time={epoch_seconds:.2f}s"
        )

    model.load_state_dict(best_state_dict)
    return model, pd.DataFrame(history)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """在测试集上评估模型。"""
    return run_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=None,
        device=device,
    )


def summarize_results(histories: dict[str, pd.DataFrame], test_results: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """整理每种激活函数的关键结果。"""
    summary_rows = []

    for activation_name, history_df in histories.items():
        best_row = history_df.loc[history_df["valid_accuracy"].idxmax()]
        test_loss, test_acc = test_results[activation_name]

        summary_rows.append(
            {
                "Activation": activation_name,
                "Best Valid Accuracy": best_row["valid_accuracy"],
                "Epoch of Best Valid Accuracy": int(best_row["epoch"]),
                "Final Train Loss": history_df.iloc[-1]["train_loss"],
                "Final Valid Loss": history_df.iloc[-1]["valid_loss"],
                "Final Valid Accuracy": history_df.iloc[-1]["valid_accuracy"],
                "Test Accuracy": test_acc,
                "Test Loss": test_loss,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "Best Valid Accuracy", ascending=False
    ).reset_index(drop=True)
    return summary_df


def plot_comparison(histories: dict[str, pd.DataFrame]) -> None:
    """绘制激活函数对比曲线。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for activation_name, history_df in histories.items():
        axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label=activation_name)
        axes[1].plot(history_df["epoch"], history_df["valid_loss"], marker="o", label=activation_name)
        axes[2].plot(history_df["epoch"], history_df["valid_accuracy"], marker="o", label=activation_name)

    axes[0].set_title("不同激活函数的训练损失对比")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].legend()

    axes[1].set_title("不同激活函数的验证损失对比")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Valid Loss")
    axes[1].legend()

    axes[2].set_title("不同激活函数的验证准确率对比")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Valid Accuracy")
    axes[2].set_ylim(0, 1.0)
    axes[2].legend()

    plt.tight_layout()
    save_plot("task2_activation_comparison_curves.png")


def main() -> None:
    # 第一步：固定随机种子并确定运行设备
    set_seed(RANDOM_SEED)
    device = get_device()
    data_root = Path(__file__).resolve().parent / "fashion_mnist"

    # 第二步：加载同一份数据，保证三组实验对比公平
    train_loader, valid_loader, test_loader = build_dataloaders(data_root)
    criterion = nn.CrossEntropyLoss()

    print("当前设备：", device)
    print("数据目录：", data_root)
    print("对比激活函数：", list(ACTIVATION_FACTORIES.keys()))

    histories = {}
    test_results = {}

    # 第三步：依次训练 ReLU、Sigmoid、Tanh 三组模型
    for activation_name, activation_cls in ACTIVATION_FACTORIES.items():
        model, history_df = train_model(
            activation_name=activation_name,
            activation_cls=activation_cls,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device,
        )
        histories[activation_name] = history_df
        test_results[activation_name] = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

    # 第四步：整理结果并输出比较表
    summary_df = summarize_results(histories, test_results)

    print("\n各激活函数完整训练记录：")
    all_history_df = pd.concat(histories.values(), ignore_index=True)
    print(all_history_df.to_string(index=False))

    print("\n激活函数对比结果汇总：")
    print(summary_df.to_string(index=False))

    # 第五步：画出曲线，方便比较收敛速度、验证精度和稳定性
    plot_comparison(histories)


if __name__ == "__main__":
    main()
