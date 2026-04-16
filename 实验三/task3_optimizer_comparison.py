#!/usr/bin/env python
# coding: utf-8

# 任务3：比较不同优化算法的影响

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
HIDDEN_DIM = 256
VALID_RATIO = 0.1

# 任务要求对比的四种优化器
# 学习率先按常见设置给出，保证比较时只改变优化策略本身
OPTIMIZER_FACTORIES = {
    "SGD": lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
    "SGD_Nesterov": lambda params: torch.optim.SGD(
        params, lr=0.01, momentum=0.9, nesterov=True
    ),
    "Adam": lambda params: torch.optim.Adam(params, lr=0.001),
    "RMSprop": lambda params: torch.optim.RMSprop(params, lr=0.001),
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


class OptimizerMLP(nn.Module):
    """一层隐藏层的 MLP，固定使用 ReLU。"""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_classes: int = 10) -> None:
        super().__init__()
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
    optimizer_name: str,
    optimizer_factory,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame]:
    """训练某一种优化器对应的模型，并记录每轮结果。"""
    # 每组实验都重新固定随机种子，尽量让对比更公平
    set_seed(RANDOM_SEED)
    model = OptimizerMLP().to(device)
    optimizer = optimizer_factory(model.parameters())

    history = []
    # 始终保留验证集效果最好的那一轮参数
    best_state_dict = copy.deepcopy(model.state_dict())
    best_valid_acc = 0.0

    print(f"\n========== 开始训练：{optimizer_name} ==========")

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
                "optimizer": optimizer_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_acc,
                "seconds": epoch_seconds,
            }
        )

        print(
            f"{optimizer_name} | Epoch {epoch:02d}/{NUM_EPOCHS} | "
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
    """整理每种优化器的关键结果。"""
    summary_rows = []

    for optimizer_name, history_df in histories.items():
        # 先找出验证集表现最好的一轮，再结合测试集结果汇总
        best_row = history_df.loc[history_df["valid_accuracy"].idxmax()]
        test_loss, test_acc = test_results[optimizer_name]

        summary_rows.append(
            {
                "Optimizer": optimizer_name,
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
    """绘制优化器对比曲线。"""
    # 分别比较训练损失、验证损失和验证准确率
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for optimizer_name, history_df in histories.items():
        axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label=optimizer_name)
        axes[1].plot(history_df["epoch"], history_df["valid_loss"], marker="o", label=optimizer_name)
        axes[2].plot(history_df["epoch"], history_df["valid_accuracy"], marker="o", label=optimizer_name)

    axes[0].set_title("不同优化器的训练损失对比")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].legend()

    axes[1].set_title("不同优化器的验证损失对比")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Valid Loss")
    axes[1].legend()

    axes[2].set_title("不同优化器的验证准确率对比")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Valid Accuracy")
    axes[2].set_ylim(0, 1.0)
    axes[2].legend()

    plt.tight_layout()
    save_plot("task3_optimizer_comparison_curves.png")


def main() -> None:
    # 第一步：固定随机种子并确定运行设备
    set_seed(RANDOM_SEED)
    device = get_device()
    data_root = Path(__file__).resolve().parent / "fashion_mnist"

    # 第二步：加载同一份数据，保证四组实验对比公平
    train_loader, valid_loader, test_loader = build_dataloaders(data_root)
    criterion = nn.CrossEntropyLoss()

    print("当前设备：", device)
    print("数据目录：", data_root)
    print("对比优化器：", list(OPTIMIZER_FACTORIES.keys()))

    histories = {}
    test_results = {}

    # 第三步：依次训练四组优化器
    for optimizer_name, optimizer_factory in OPTIMIZER_FACTORIES.items():
        model, history_df = train_model(
            optimizer_name=optimizer_name,
            optimizer_factory=optimizer_factory,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device,
        )
        histories[optimizer_name] = history_df
        test_results[optimizer_name] = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

    # 第四步：整理结果并输出比较表
    summary_df = summarize_results(histories, test_results)

    print("\n各优化器完整训练记录：")
    all_history_df = pd.concat(histories.values(), ignore_index=True)
    print(all_history_df.to_string(index=False))

    print("\n优化器对比结果汇总：")
    print(summary_df.to_string(index=False))

    # 第五步：画出曲线，比较收敛速度、验证精度和稳定性
    plot_comparison(histories)


if __name__ == "__main__":
    main()
