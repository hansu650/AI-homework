#!/usr/bin/env python
# coding: utf-8

# 任务4：过拟合与正则化实践

import copy
import random
import time
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
VALID_RATIO = 0.1
REDUCED_TRAIN_RATIO = 0.1
DROPOUT_P = 0.5

EXPERIMENTS = {
    "No_Regularization": {"use_dropout": False},
    "Dropout_p0.5": {"use_dropout": True},
}


def set_seed(seed: int) -> None:
    """固定随机种子，尽量让实验结果更稳定。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """优先使用 GPU，没有 GPU 就退回 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_plot(filename: str) -> None:
    """保存图像，在交互环境下顺便显示。"""
    output_path = Path(__file__).resolve().parent / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if "agg" in plt.get_backend().lower():
        plt.close()
    else:
        plt.show()


class DeepMLP(nn.Module):
    """更深的多层感知机，可选是否加入 Dropout。"""

    def __init__(self, use_dropout: bool = False, dropout_p: float = DROPOUT_P) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
        ]
        if use_dropout:
            layers.append(nn.Dropout(p=dropout_p))

        layers.extend(
            [
                nn.Linear(512, 256),
                nn.ReLU(),
            ]
        )
        if use_dropout:
            layers.append(nn.Dropout(p=dropout_p))

        layers.extend(
            [
                nn.Linear(256, 128),
                nn.ReLU(),
            ]
        )
        if use_dropout:
            layers.append(nn.Dropout(p=dropout_p))

        layers.append(nn.Linear(128, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_dataloaders(data_root: Path):
    """先划验证集，再从训练池里抽取 10% 做真正训练集。"""
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
    train_pool_size = len(train_full) - valid_size

    split_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_pool, valid_dataset = random_split(
        train_full,
        [train_pool_size, valid_size],
        generator=split_generator,
    )

    reduced_train_size = max(1, int(len(train_pool) * REDUCED_TRAIN_RATIO))
    remaining_size = len(train_pool) - reduced_train_size
    subset_generator = torch.Generator().manual_seed(RANDOM_SEED)
    reduced_train_dataset, _ = random_split(
        train_pool,
        [reduced_train_size, remaining_size],
        generator=subset_generator,
    )

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(reduced_train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, valid_loader, test_loader, reduced_train_size, len(valid_dataset)


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, float]:
    """执行一轮训练或验证。"""
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
    experiment_name: str,
    use_dropout: bool,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame]:
    """训练某一组实验并记录每轮结果。"""
    set_seed(RANDOM_SEED)

    model = DeepMLP(use_dropout=use_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_state_dict = copy.deepcopy(model.state_dict())
    best_valid_acc = 0.0

    print(f"\n========== 开始训练：{experiment_name} ==========")

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
                "experiment": experiment_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_acc,
                "accuracy_gap": train_acc - valid_acc,
                "seconds": epoch_seconds,
            }
        )

        print(
            f"{experiment_name} | Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f} | "
            f"gap={train_acc - valid_acc:.4f} | "
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


def summarize_results(
    histories: dict[str, pd.DataFrame],
    test_results: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """整理每组实验的关键结果。"""
    summary_rows = []

    for experiment_name, history_df in histories.items():
        best_row = history_df.loc[history_df["valid_accuracy"].idxmax()]
        test_loss, test_acc = test_results[experiment_name]

        summary_rows.append(
            {
                "Experiment": experiment_name,
                "Best Valid Accuracy": best_row["valid_accuracy"],
                "Epoch of Best Valid Accuracy": int(best_row["epoch"]),
                "Final Train Accuracy": history_df.iloc[-1]["train_accuracy"],
                "Final Valid Accuracy": history_df.iloc[-1]["valid_accuracy"],
                "Final Accuracy Gap": history_df.iloc[-1]["accuracy_gap"],
                "Test Accuracy": test_acc,
                "Test Loss": test_loss,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "Best Valid Accuracy", ascending=False
    ).reset_index(drop=True)
    return summary_df


def plot_comparison(histories: dict[str, pd.DataFrame]) -> None:
    """绘制过拟合与正则化对比曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for experiment_name, history_df in histories.items():
        axes[0, 0].plot(
            history_df["epoch"], history_df["train_loss"], marker="o", label=experiment_name
        )
        axes[0, 1].plot(
            history_df["epoch"], history_df["valid_loss"], marker="o", label=experiment_name
        )
        axes[1, 0].plot(
            history_df["epoch"], history_df["valid_accuracy"], marker="o", label=experiment_name
        )
        axes[1, 1].plot(
            history_df["epoch"], history_df["accuracy_gap"], marker="o", label=experiment_name
        )

    axes[0, 0].set_title("训练损失对比")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Train Loss")
    axes[0, 0].legend()

    axes[0, 1].set_title("验证损失对比")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Valid Loss")
    axes[0, 1].legend()

    axes[1, 0].set_title("验证准确率对比")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Valid Accuracy")
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].legend()

    axes[1, 1].set_title("训练/验证准确率间隙对比")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy Gap")
    axes[1, 1].legend()

    plt.tight_layout()
    save_plot("task4_overfitting_regularization_curves.png")


def main() -> None:
    # 第一步：固定随机种子并确定运行设备
    set_seed(RANDOM_SEED)
    device = get_device()
    data_root = Path(__file__).resolve().parent / "fashion_mnist"

    # 第二步：准备小训练集，让模型更容易出现过拟合
    train_loader, valid_loader, test_loader, train_subset_size, valid_size = build_dataloaders(
        data_root
    )
    criterion = nn.CrossEntropyLoss()

    print("当前设备：", device)
    print("数据目录：", data_root)
    print("训练子集样本数：", train_subset_size)
    print("验证集样本数：", valid_size)
    print("对比实验：", list(EXPERIMENTS.keys()))

    histories = {}
    test_results = {}

    # 第三步：先训练无正则化模型，再训练加入 Dropout 的模型
    for experiment_name, config in EXPERIMENTS.items():
        model, history_df = train_model(
            experiment_name=experiment_name,
            use_dropout=config["use_dropout"],
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device,
        )
        histories[experiment_name] = history_df
        test_results[experiment_name] = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

    # 第四步：整理结果并输出比较表
    summary_df = summarize_results(histories, test_results)

    print("\n完整训练记录：")
    all_history_df = pd.concat(histories.values(), ignore_index=True)
    print(all_history_df.to_string(index=False))

    print("\n结果汇总表：")
    print(summary_df.to_string(index=False))

    # 第五步：绘图观察过拟合与正则化效果
    plot_comparison(histories)


if __name__ == "__main__":
    main()
