#!/usr/bin/env python
# coding: utf-8

"""
实验4：Fashion-MNIST 卷积神经网络分类

代码按实验报告的顺序写：先读数据，再搭 CNN、训练模型，
最后把曲线、混淆矩阵、预测样例和通道数对比图都保存下来。
"""

import random
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


warnings.filterwarnings("ignore")

# 画图时尽量用中文字体，后面截图放进报告会清楚一些。
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# 这些参数后面训练和画图都会用到，集中放在这里比较好改。
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

# 通道数对比会额外训练三次；如果赶时间，可以先改成 False。
RUN_HYPERPARAMETER_COMPARISON = True
HYPERPARAM_EPOCHS = 5
HYPERPARAM_TRAIN_LIMIT = 15000

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT3_DATA_DIR = SCRIPT_DIR.parent / "实验三_发送版" / "fashion_mnist"

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


def set_seed(seed: int = RANDOM_SEED) -> None:
    """把几个随机源固定住，减少每次运行之间的波动。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """有 CUDA 就用显卡，没有就退回 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def has_fashion_mnist_files(data_root: Path) -> bool:
    """检查 root/FashionMNIST/raw 里是不是已经有数据文件。"""
    raw_dir = data_root / "FashionMNIST" / "raw"
    required_files = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    return raw_dir.exists() and all((raw_dir / name).exists() for name in required_files)


def resolve_data_root() -> Path:
    """先找实验四自己的数据；没有的话直接用实验三那份。"""
    candidates = [
        SCRIPT_DIR / "fashion_mnist",
        EXPERIMENT3_DATA_DIR,
    ]

    for data_root in candidates:
        if has_fashion_mnist_files(data_root):
            return data_root

    message = (
        "未找到 Fashion-MNIST 本地数据集。\n"
        f"已检查：\n"
        f"1. {candidates[0]}\n"
        f"2. {candidates[1]}\n"
        "请将实验三的 fashion_mnist 文件夹复制到实验四目录，"
        "或在确认可以联网后把 datasets.FashionMNIST 的 download=False 改成 download=True。"
    )
    raise FileNotFoundError(message)


def save_plot(filename: str) -> None:
    """图统一存到实验四目录，终端里也顺便提示一下。"""
    output_path = SCRIPT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图像已保存：{filename}")


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """显示图片前，把 [-1, 1] 的张量拉回 [0, 1]。"""
    return image * 0.5 + 0.5


def build_dataloaders(data_root: Path) -> Tuple[datasets.FashionMNIST, datasets.FashionMNIST, DataLoader, DataLoader]:
    """读取训练集和测试集，并打包成 DataLoader。"""
    # 这里先把图片转成张量，再按题目允许的方式归一化到 [-1, 1]。
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.FashionMNIST(
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

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_dataset, test_dataset, train_loader, test_loader


class FashionCNN(nn.Module):
    """实验中用的基础 CNN，结构和题目要求保持一致。"""

    def __init__(self, channels: Tuple[int, int] = (32, 64), num_classes: int = 10) -> None:
        super().__init__()
        c1, c2 = channels

        self.features = nn.Sequential(
            # 第一段卷积从原图里提局部特征，池化后尺寸减半。
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二段继续提特征，最后得到 c2 个 7x7 特征图。
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    """训练和测试都走这里；传 optimizer 时才会反向更新。"""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    log_prefix: str = "",
) -> List[Dict[str, float]]:
    """按 epoch 训练模型，把后面画图要用的数据记下来。"""
    history: List[Dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = run_one_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        epoch_seconds = time.time() - start_time
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "seconds": epoch_seconds,
            }
        )

        print(
            f"{log_prefix}Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"time={epoch_seconds:.2f}s"
        )

    return history


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """把测试集预测结果攒起来，后面画混淆矩阵用。"""
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def make_confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """自己算一个混淆矩阵，行是真实类别，列是预测类别。"""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(labels, preds):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def make_subset(dataset: Dataset, limit: Optional[int], seed: int) -> Dataset:
    """超参数对比只抽一部分训练样本，节省一点运行时间。"""
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def get_targets(dataset: datasets.FashionMNIST) -> np.ndarray:
    """拿到数据集标签，用来统计每一类有多少张。"""
    return np.array(dataset.targets)


def plot_sample_images(dataset: datasets.FashionMNIST, filename: str = "exp4_sample_images.png") -> None:
    """随机挑一些训练图片，看一下数据大概长什么样。"""
    rng = random.Random(RANDOM_SEED)
    indices = rng.sample(range(len(dataset)), 16)

    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for ax, idx in zip(axes.flat, indices):
        image, label = dataset[idx]
        image = denormalize_image(image).squeeze(0)
        ax.imshow(image, cmap="gray")
        ax.set_title(CLASS_NAMES[label], fontsize=10)
        ax.axis("off")

    plt.suptitle("Fashion-MNIST 样本图像", fontsize=15)
    plt.tight_layout()
    save_plot(filename)


def plot_class_distribution(
    train_dataset: datasets.FashionMNIST,
    test_dataset: datasets.FashionMNIST,
    filename: str = "exp4_class_distribution.png",
) -> None:
    """看训练集和测试集的类别分布是否均衡。"""
    train_counts = np.bincount(get_targets(train_dataset), minlength=len(CLASS_NAMES))
    test_counts = np.bincount(get_targets(test_dataset), minlength=len(CLASS_NAMES))

    x = np.arange(len(CLASS_NAMES))
    width = 0.38
    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, train_counts, width=width, label="训练集")
    plt.bar(x + width / 2, test_counts, width=width, label="测试集")
    plt.xticks(x, CLASS_NAMES, rotation=35, ha="right")
    plt.ylabel("样本数量")
    plt.title("Fashion-MNIST 各类别样本分布")
    plt.legend()
    plt.tight_layout()
    save_plot(filename)


def history_values(history: Sequence[Dict[str, float]], key: str) -> List[float]:
    return [item[key] for item in history]


def plot_training_loss(history: Sequence[Dict[str, float]], filename: str = "exp4_training_loss.png") -> None:
    """把 loss 曲线单独画出来，报告里可以直接放。"""
    epochs = history_values(history, "epoch")
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_values(history, "train_loss"), marker="o", label="训练损失")
    plt.plot(epochs, history_values(history, "test_loss"), marker="s", label="测试损失")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN 训练损失变化曲线")
    plt.xticks(epochs)
    plt.legend()
    plt.tight_layout()
    save_plot(filename)


def plot_accuracy_curve(history: Sequence[Dict[str, float]], filename: str = "exp4_accuracy_curve.png") -> None:
    """把训练和测试准确率放在一张图里比较。"""
    epochs = history_values(history, "epoch")
    train_acc = [acc * 100 for acc in history_values(history, "train_acc")]
    test_acc = [acc * 100 for acc in history_values(history, "test_acc")]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, marker="o", label="训练准确率")
    plt.plot(epochs, test_acc, marker="s", label="测试准确率")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CNN 准确率变化曲线")
    plt.xticks(epochs)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    save_plot(filename)


def plot_epoch_time(history: Sequence[Dict[str, float]], filename: str = "exp4_epoch_time.png") -> None:
    """记录每轮耗时，顺手看一下运行速度是否稳定。"""
    epochs = history_values(history, "epoch")
    seconds = history_values(history, "seconds")

    plt.figure(figsize=(8, 5))
    plt.bar(epochs, seconds, color="#4C78A8")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("每轮训练耗时")
    plt.xticks(epochs)
    plt.tight_layout()
    save_plot(filename)


def plot_confusion_matrix(
    matrix: np.ndarray,
    filename: str = "exp4_confusion_matrix.png",
    normalized: bool = False,
) -> None:
    """画混淆矩阵，重点看哪些衣服类别容易分错。"""
    if normalized:
        row_sums = matrix.sum(axis=1, keepdims=True)
        data = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0) * 100
        fmt = ".1f"
        title = "Fashion-MNIST 测试集混淆矩阵（百分比）"
        cbar_label = "百分比 (%)"
    else:
        data = matrix
        fmt = "d"
        title = "Fashion-MNIST 测试集混淆矩阵"
        cbar_label = "样本数量"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": cbar_label},
    )
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_plot(filename)


def plot_per_class_accuracy(matrix: np.ndarray, filename: str = "exp4_per_class_accuracy.png") -> None:
    """每一类单独算准确率，比只看总准确率更细一点。"""
    row_sums = matrix.sum(axis=1)
    class_acc = np.divide(np.diag(matrix), row_sums, out=np.zeros_like(row_sums, dtype=float), where=row_sums != 0)

    plt.figure(figsize=(11, 5))
    bars = plt.bar(CLASS_NAMES, class_acc * 100, color="#59A14F")
    plt.ylabel("Accuracy (%)")
    plt.title("各类别测试准确率")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 100)

    for bar, acc in zip(bars, class_acc):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    save_plot(filename)


def plot_prediction_examples(
    model: nn.Module,
    dataset: datasets.FashionMNIST,
    device: torch.device,
    filename: str = "exp4_prediction_examples.png",
) -> None:
    """从测试集随机挑 8 张，直观看预测是否靠谱。"""
    rng = random.Random(RANDOM_SEED + 7)
    indices = rng.sample(range(len(dataset)), 8)

    images = torch.stack([dataset[idx][0] for idx in indices])
    labels = torch.tensor([dataset[idx][1] for idx in indices])

    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, image, true_label, pred_label in zip(axes.flat, images, labels, preds):
        image = denormalize_image(image).squeeze(0)
        ax.imshow(image, cmap="gray")
        color = "green" if int(true_label) == int(pred_label) else "red"
        ax.set_title(
            f"True: {CLASS_NAMES[int(true_label)]}\nPred: {CLASS_NAMES[int(pred_label)]}",
            color=color,
            fontsize=10,
        )
        ax.axis("off")

    plt.suptitle("测试集随机预测结果展示", fontsize=15)
    plt.tight_layout()
    save_plot(filename)


def plot_hyperparameter_comparison(
    results: Sequence[Dict[str, float]],
    filename: str = "exp4_hyperparameter_comparison.png",
) -> None:
    """把 small、base、large 三种通道数放在一起比较。"""
    names = [item["name"] for item in results]
    test_acc = [item["test_acc"] * 100 for item in results]
    train_acc = [item["train_acc"] * 100 for item in results]
    seconds = [item["seconds"] for item in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(names))
    width = 0.35
    axes[0].bar(x - width / 2, train_acc, width=width, label="训练准确率")
    axes[0].bar(x + width / 2, test_acc, width=width, label="测试准确率")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("不同卷积通道数的准确率对比")
    axes[0].set_ylim(0, 100)
    axes[0].legend()

    axes[1].bar(names, seconds, color="#F28E2B")
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("不同卷积通道数的训练耗时")

    plt.tight_layout()
    save_plot(filename)


def print_per_class_accuracy(matrix: np.ndarray) -> None:
    """终端也打印一份每类准确率，方便截图或写分析。"""
    row_sums = matrix.sum(axis=1)
    class_acc = np.divide(np.diag(matrix), row_sums, out=np.zeros_like(row_sums, dtype=float), where=row_sums != 0)

    print("\n各类别测试准确率：")
    for class_name, acc in zip(CLASS_NAMES, class_acc):
        print(f"{class_name:12s}: {acc * 100:.2f}%")


def print_top_confusions(matrix: np.ndarray, top_k: int = 5) -> None:
    """把错得最多的类别对列出来，写报告时很有用。"""
    confusion_pairs = []
    for true_idx in range(matrix.shape[0]):
        for pred_idx in range(matrix.shape[1]):
            if true_idx != pred_idx:
                confusion_pairs.append((matrix[true_idx, pred_idx], true_idx, pred_idx))

    confusion_pairs.sort(reverse=True)
    print(f"\n混淆最多的前 {top_k} 个类别对：")
    for count, true_idx, pred_idx in confusion_pairs[:top_k]:
        print(f"真实 {CLASS_NAMES[true_idx]:12s} -> 预测 {CLASS_NAMES[pred_idx]:12s}: {count} 张")


def run_hyperparameter_comparison(
    train_dataset: datasets.FashionMNIST,
    test_loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, float]]:
    """简单看一下通道数变多后，准确率和耗时怎么变。"""
    print("\n========== 超参数对比：不同卷积通道数 ==========")
    print(
        f"说明：为节省时间，超参数对比每组训练 {HYPERPARAM_EPOCHS} epochs，"
        f"训练子集样本数为 {min(HYPERPARAM_TRAIN_LIMIT, len(train_dataset))}。"
    )

    train_subset = make_subset(train_dataset, HYPERPARAM_TRAIN_LIMIT, RANDOM_SEED + 99)
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    experiments = [
        ("small 16-32", (16, 32)),
        ("base 32-64", (32, 64)),
        ("large 64-128", (64, 128)),
    ]

    criterion = nn.CrossEntropyLoss()
    results: List[Dict[str, float]] = []

    for name, channels in experiments:
        set_seed(RANDOM_SEED)
        model = FashionCNN(channels=channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        start_time = time.time()
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=HYPERPARAM_EPOCHS,
            log_prefix=f"{name} | ",
        )
        total_seconds = time.time() - start_time
        final_row = history[-1]
        results.append(
            {
                "name": name,
                "train_acc": final_row["train_acc"],
                "test_acc": final_row["test_acc"],
                "seconds": total_seconds,
            }
        )

    print("\n超参数对比结果汇总：")
    for item in results:
        print(
            f"{item['name']:12s} | "
            f"train_acc={item['train_acc']:.4f} | "
            f"test_acc={item['test_acc']:.4f} | "
            f"time={item['seconds']:.2f}s"
        )

    plot_hyperparameter_comparison(results)
    return results


def main() -> None:
    set_seed(RANDOM_SEED)

    device = get_device()
    data_root = resolve_data_root()
    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(data_root)

    print("当前设备：", device)
    print("数据目录：", data_root)
    print(f"训练集样本数：{len(train_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")
    print(f"batch size：{BATCH_SIZE}")
    print(f"epochs：{NUM_EPOCHS}")
    print(f"学习率：{LEARNING_RATE}")
    print("类别名称：", CLASS_NAMES)

    # 先保存数据集图片，报告“数据集准备”部分可以直接用。
    plot_sample_images(train_dataset)
    plot_class_distribution(train_dataset, test_dataset)

    model = FashionCNN(channels=(32, 64)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n模型结构：")
    print(model)

    print("\n========== 开始训练基础 CNN ==========")
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
    )

    final_test_loss, final_test_acc = run_one_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
    )
    print("\n最终测试集结果：")
    print(f"test_loss={final_test_loss:.4f}, test_accuracy={final_test_acc:.4f}")
    print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")

    labels, preds = collect_predictions(model, test_loader, device)
    matrix = make_confusion_matrix(labels, preds, num_classes=len(CLASS_NAMES))

    print_per_class_accuracy(matrix)
    print_top_confusions(matrix)

    # 训练结束后，把主要结果图一次性保存下来。
    plot_training_loss(history)
    plot_accuracy_curve(history)
    plot_epoch_time(history)
    plot_confusion_matrix(matrix, filename="exp4_confusion_matrix.png", normalized=False)
    plot_confusion_matrix(matrix, filename="exp4_confusion_matrix_normalized.png", normalized=True)
    plot_per_class_accuracy(matrix)
    plot_prediction_examples(model, test_dataset, device)

    if RUN_HYPERPARAMETER_COMPARISON:
        run_hyperparameter_comparison(train_dataset, test_loader, device)
    else:
        print("\n已跳过超参数对比。如需生成 exp4_hyperparameter_comparison.png，")
        print("请将 RUN_HYPERPARAMETER_COMPARISON 改为 True 后重新运行。")

    print("\n实验4程序运行结束。所有图像均已保存到实验四目录。")


if __name__ == "__main__":
    main()
