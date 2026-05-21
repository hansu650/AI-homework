#!/usr/bin/env python
# coding: utf-8

"""
实验六：生成式人工智能基础实验

使用本地 MNIST 压缩数据训练一个 DCGAN 风格的 GAN，并保存训练曲线、
判别器分数曲线、生成图片、潜在空间插值图、训练历史和模型权重。
"""

import argparse
import gzip
import json
import random
import struct
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


warnings.filterwarnings("ignore")

# Windows 下重定向日志时也尽量保持中文输出稳定。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

# 图表中文字体设置，避免保存图片时标题和坐标轴乱码。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 参数集中区，后续实验调参主要改这里或命令行参数。
# =========================
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.0002
LATENT_DIM = 100
SAMPLE_EVERY = 5
NUM_WORKERS = 0
FIXED_SAMPLE_COUNT = 64

SCRIPT_DIR = Path(__file__).resolve().parent
MNIST_DIR = SCRIPT_DIR / "MNIST"
OUTPUT_DIR = SCRIPT_DIR / "outputs_exp6"

LOCAL_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def print_header(title: str) -> None:
    print(f"\n========== {title} ==========")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """固定随机种子，让快速测试和正式训练都尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """自动选择 GPU/CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_idx_images(path: Path) -> np.ndarray:
    """读取 MNIST IDX 图像 gzip 文件，返回形状为 [N, 28, 28] 的 uint8 数组。"""
    with gzip.open(path, "rb") as file:
        magic, num_images, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"{path} 不是合法的 IDX 图像文件，magic={magic}")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    return data.reshape(num_images, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    """读取 MNIST IDX 标签 gzip 文件，返回形状为 [N] 的 int64 数组。"""
    with gzip.open(path, "rb") as file:
        magic, num_labels = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"{path} 不是合法的 IDX 标签文件，magic={magic}")
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    if len(labels) != num_labels:
        raise ValueError(f"{path} 标签数量不一致，文件头={num_labels}，实际={len(labels)}")
    return labels.astype(np.int64)


def normalize_mnist_images(images: np.ndarray) -> np.ndarray:
    """将像素从 [0, 255] 归一化到 [-1, 1]，并补上通道维度。"""
    images = images.astype(np.float32) / 127.5 - 1.0
    return images[:, None, :, :]


def load_local_mnist(mnist_dir: Path) -> Dict[str, np.ndarray]:
    """优先读取实验六/MNIST 下的本地 gzip 数据，不强制联网下载。"""
    required_paths = {key: mnist_dir / name for key, name in LOCAL_MNIST_FILES.items()}
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("本地 MNIST 文件不完整：" + "；".join(missing))

    train_images = read_idx_images(required_paths["train_images"])
    train_labels = read_idx_labels(required_paths["train_labels"])
    test_images = read_idx_images(required_paths["test_images"])
    test_labels = read_idx_labels(required_paths["test_labels"])

    if len(train_images) != len(train_labels):
        raise ValueError("训练图像和训练标签数量不一致。")
    if len(test_images) != len(test_labels):
        raise ValueError("测试图像和测试标签数量不一致。")

    return {
        "train_images": normalize_mnist_images(train_images),
        "train_labels": train_labels,
        "test_images": normalize_mnist_images(test_images),
        "test_labels": test_labels,
        "source": np.array([f"本地数据：{mnist_dir}"]),
    }


def load_torchvision_mnist() -> Dict[str, np.ndarray]:
    """本地数据不存在时才 fallback 到 torchvision.datasets.MNIST(download=True)。"""
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError("本地 MNIST 不存在，且未安装 torchvision，无法自动下载 MNIST。") from exc

    data_root = SCRIPT_DIR / "mnist_torchvision_cache"
    train_dataset = datasets.MNIST(root=str(data_root), train=True, download=True)
    test_dataset = datasets.MNIST(root=str(data_root), train=False, download=True)

    return {
        "train_images": normalize_mnist_images(train_dataset.data.numpy()),
        "train_labels": train_dataset.targets.numpy().astype(np.int64),
        "test_images": normalize_mnist_images(test_dataset.data.numpy()),
        "test_labels": test_dataset.targets.numpy().astype(np.int64),
        "source": np.array([f"torchvision 下载缓存：{data_root}"]),
    }


def resolve_mnist_data(mnist_dir: Path = MNIST_DIR) -> Dict[str, np.ndarray]:
    """先用本地 MNIST；如果缺失，再尝试 torchvision 下载。"""
    try:
        data = load_local_mnist(mnist_dir)
        print("数据来源：使用实验六/MNIST 本地 gzip 文件。")
        return data
    except FileNotFoundError as exc:
        print(f"本地 MNIST 不完整，准备 fallback 到 torchvision：{exc}")
        data = load_torchvision_mnist()
        print("数据来源：使用 torchvision.datasets.MNIST(download=True)。")
        return data


class MNISTTensorDataset(Dataset):
    """把已经归一化到 [-1, 1] 的 MNIST 数组包装成 Dataset。"""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def make_loader(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = BATCH_SIZE,
    train_limit: Optional[int] = None,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, Dataset]:
    """构造训练 DataLoader；train_limit 用于快速跑通代码。"""
    dataset: Dataset = MNISTTensorDataset(images, labels)
    if train_limit is not None and train_limit > 0 and train_limit < len(dataset):
        generator = torch.Generator().manual_seed(RANDOM_SEED)
        indices = torch.randperm(len(dataset), generator=generator)[:train_limit].tolist()
        dataset = Subset(dataset, indices)

    if len(dataset) < 2:
        raise ValueError("训练样本数太少，至少需要 2 张图片才能训练带 BatchNorm 的 GAN。")

    drop_last = len(dataset) >= batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset


class Generator(nn.Module):
    """DCGAN 风格生成器：噪声向量 -> 1x28x28 的手写数字图像。"""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(z.size(0), 128, 7, 7)
        return self.deconv(x)


class Discriminator(nn.Module):
    """DCGAN 风格判别器：判断输入图像是真实 MNIST 图片的概率。"""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 这里返回 logit，并配合 BCEWithLogitsLoss 使用；等价于 Sigmoid + BCELoss，
        # 但数值更稳定。torch.sigmoid(logit) 才是“输入图像为真实图片”的概率。
        return self.classifier(self.features(x))


def discriminator_probability(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_image_grid(
    images: torch.Tensor,
    output_path: Path,
    title: str,
    nrow: int = 8,
    max_images: int = 64,
) -> None:
    """不依赖 torchvision.utils.make_grid，直接用 matplotlib 保存灰度图片网格。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu()[:max_images]
    images = ((images + 1.0) / 2.0).clamp(0.0, 1.0)

    cols = min(nrow, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.25))
    axes_array = np.array(axes).reshape(-1)

    for ax in axes_array:
        ax.axis("off")
    for ax, image in zip(axes_array, images):
        ax.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    save_plot(output_path)


def generate_from_fixed_noise(
    generator: Generator,
    fixed_noise: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """用固定噪声生成图片，用于观察同一组噪声随 epoch 的变化。"""
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        generated = generator(fixed_noise.to(device)).cpu()
    generator.train(was_training)
    return generated


def plot_loss_curve(history: Sequence[Dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["D_loss"] for row in history], marker="o", label="D_loss")
    plt.plot(epochs, [row["G_loss"] for row in history], marker="s", label="G_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN 训练损失曲线")
    plt.legend()
    plt.grid(alpha=0.25)
    save_plot(output_path)


def plot_score_curve(history: Sequence[Dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["D_x"] for row in history], marker="o", label="D(x)：真实图像判真概率")
    plt.plot(epochs, [row["D_G_z"] for row in history], marker="s", label="D(G(z))：生成图像判真概率")
    plt.xlabel("Epoch")
    plt.ylabel("判别器平均输出概率")
    plt.title("判别器分数变化曲线")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.25)
    save_plot(output_path)


def save_generation_progress(
    saved_samples: Dict[int, torch.Tensor],
    output_path: Path,
    preferred_epochs: Sequence[int] = (1, 10, 20, 30, 40, 50),
) -> None:
    """拼接固定噪声在不同 epoch 的生成效果，默认展示 1/10/20/30/40/50。"""
    if not saved_samples:
        return

    available_epochs = sorted(saved_samples)
    chosen_epochs = [epoch for epoch in preferred_epochs if epoch in saved_samples]
    for epoch in available_epochs:
        if epoch not in chosen_epochs:
            chosen_epochs.append(epoch)
        if len(chosen_epochs) >= min(6, len(available_epochs)):
            break
    chosen_epochs = sorted(chosen_epochs[:6])

    cols = 8
    rows = len(chosen_epochs)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.35))
    axes = np.array(axes).reshape(rows, cols)

    for row_idx, epoch in enumerate(chosen_epochs):
        images = ((saved_samples[epoch][:cols] + 1.0) / 2.0).clamp(0.0, 1.0)
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            ax.imshow(images[col_idx].squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
            if col_idx == 0:
                ax.set_title(f"Epoch {epoch}", fontsize=10)

    fig.suptitle("固定噪声生成效果对比", fontsize=14)
    plt.tight_layout()
    save_plot(output_path)


def save_latent_interpolation(
    generator: Generator,
    latent_dim: int,
    device: torch.device,
    output_path: Path,
    steps: int = 10,
) -> None:
    """在两个随机噪声向量之间插值，观察潜在空间的平滑变化。"""
    generator.eval()
    z_start = torch.randn(1, latent_dim, device=device)
    z_end = torch.randn(1, latent_dim, device=device)
    ratios = torch.linspace(0, 1, steps, device=device).view(steps, 1)
    z = z_start * (1 - ratios) + z_end * ratios

    with torch.no_grad():
        images = generator(z).cpu()
    save_image_grid(images, output_path, title="潜在空间线性插值", nrow=steps, max_images=steps)


def train_gan(
    train_loader: DataLoader,
    generator: Generator,
    discriminator: Discriminator,
    device: torch.device,
    epochs: int,
    latent_dim: int,
    lr: float,
    sample_every: int,
    output_dir: Path,
) -> Tuple[List[Dict[str, float]], Dict[int, torch.Tensor], torch.Tensor]:
    """按 batch 交替训练 D 和 G，并保存固定噪声的生成结果。"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    fixed_noise = torch.randn(FIXED_SAMPLE_COUNT, latent_dim, device=device)

    history: List[Dict[str, float]] = []
    saved_samples: Dict[int, torch.Tensor] = {}

    print_header("开始训练 GAN")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        totals = {
            "D_loss": 0.0,
            "G_loss": 0.0,
            "D_loss_real": 0.0,
            "D_loss_fake": 0.0,
            "D_x": 0.0,
            "D_G_z": 0.0,
        }
        sample_count = 0

        generator.train()
        discriminator.train()

        for real_images, _ in train_loader:
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)
            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            # 第一步：训练判别器 D，让它区分真实图像和生成图像。
            optimizer_d.zero_grad()
            real_logits = discriminator(real_images)
            d_loss_real = criterion(real_logits, real_targets)

            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            fake_logits_for_d = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_logits_for_d, fake_targets)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # 第二步：训练生成器 G，让生成图像尽量骗过判别器 D。
            optimizer_g.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            fake_logits_for_g = discriminator(fake_images)
            g_loss = criterion(fake_logits_for_g, real_targets)
            g_loss.backward()
            optimizer_g.step()

            with torch.no_grad():
                d_x = discriminator_probability(real_logits).mean().item()
                d_g_z = discriminator_probability(fake_logits_for_g).mean().item()

            totals["D_loss"] += d_loss.item() * batch_size
            totals["G_loss"] += g_loss.item() * batch_size
            totals["D_loss_real"] += d_loss_real.item() * batch_size
            totals["D_loss_fake"] += d_loss_fake.item() * batch_size
            totals["D_x"] += d_x * batch_size
            totals["D_G_z"] += d_g_z * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise RuntimeError("没有可训练 batch，请调大 --train-limit 或调小 --batch-size。")

        row = {key: value / sample_count for key, value in totals.items()}
        row["epoch"] = epoch
        row["seconds"] = time.time() - start_time
        history.append(row)

        print(
            f"第 {epoch:03d}/{epochs:03d} 轮 | "
            f"D_loss={row['D_loss']:.4f} | G_loss={row['G_loss']:.4f} | "
            f"D_loss_real={row['D_loss_real']:.4f} | D_loss_fake={row['D_loss_fake']:.4f} | "
            f"D(x)={row['D_x']:.4f} | D(G(z))={row['D_G_z']:.4f} | "
            f"用时={row['seconds']:.2f}s"
        )

        should_save_epoch = epoch == 1 or (sample_every > 0 and epoch % sample_every == 0)
        if should_save_epoch:
            generated = generate_from_fixed_noise(generator, fixed_noise, device)
            saved_samples[epoch] = generated
            save_image_grid(
                generated,
                output_dir / f"exp6_generated_epoch_{epoch:03d}.png",
                title=f"第 {epoch} 轮生成样本",
            )

    final_samples = generate_from_fixed_noise(generator, fixed_noise, device)
    saved_samples[epochs] = final_samples
    save_image_grid(final_samples, output_dir / "exp6_generated_final.png", title="最终生成样本")
    return history, saved_samples, fixed_noise.detach().cpu()


def write_summary(
    output_path: Path,
    args: argparse.Namespace,
    data_source: str,
    device: torch.device,
    train_size: int,
    history: Sequence[Dict[str, float]],
) -> None:
    """保存纯文本摘要，方便截图或后续整理报告时引用。"""
    final_row = history[-1]
    lines = [
        "实验六：MNIST GAN 训练摘要",
        "",
        f"数据来源：{data_source}",
        f"训练样本数：{train_size}",
        f"运行设备：{device}",
        f"随机种子：{RANDOM_SEED}",
        f"Epoch：{args.epochs}",
        f"Batch Size：{args.batch_size}",
        f"学习率：{args.lr}",
        f"噪声维度：{args.latent_dim}",
        f"采样间隔：{args.sample_every}",
        "",
        "最终一轮指标：",
        f"D_loss：{final_row['D_loss']:.6f}",
        f"G_loss：{final_row['G_loss']:.6f}",
        f"D_loss_real：{final_row['D_loss_real']:.6f}",
        f"D_loss_fake：{final_row['D_loss_fake']:.6f}",
        f"D(x)：{final_row['D_x']:.6f}",
        f"D(G(z))：{final_row['D_G_z']:.6f}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验六：使用 MNIST 训练 DCGAN 风格 GAN")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="训练轮数，默认 50")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch Size，默认 64")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Adam 学习率，默认 0.0002")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM, help="随机噪声向量维度，默认 100")
    parser.add_argument("--train-limit", type=int, default=None, help="只取前若干训练样本，用于快速测试")
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY, help="每隔多少轮保存生成图，默认 5")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="DataLoader worker 数，Windows 默认 0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_header("实验六：生成式人工智能基础实验")
    print("任务：使用 MNIST 手写数字数据集训练 DCGAN 风格 GAN。")
    print(f"脚本目录：{SCRIPT_DIR}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"随机种子：{RANDOM_SEED}")

    device = get_device()
    print(f"当前设备：{device}")
    if torch.cuda.is_available():
        print(f"CUDA 设备名称：{torch.cuda.get_device_name(0)}")

    print_header("读取数据")
    data = resolve_mnist_data(MNIST_DIR)
    data_source = str(data["source"][0])
    train_images = data["train_images"]
    train_labels = data["train_labels"]
    print(f"训练集图像形状：{train_images.shape}")
    print(f"训练集标签形状：{train_labels.shape}")
    print(f"像素范围：[{train_images.min():.1f}, {train_images.max():.1f}]")

    train_loader, train_dataset = make_loader(
        train_images,
        train_labels,
        batch_size=args.batch_size,
        train_limit=args.train_limit,
        num_workers=args.num_workers,
    )
    print(f"实际训练样本数：{len(train_dataset)}")
    print(f"Batch Size：{args.batch_size}")
    print(f"Epoch：{args.epochs}")
    print(f"学习率：{args.lr}")
    print(f"噪声维度：{args.latent_dim}")

    real_sample_count = min(FIXED_SAMPLE_COUNT, len(train_images))
    real_samples = torch.from_numpy(train_images[:real_sample_count]).float()
    save_image_grid(real_samples, OUTPUT_DIR / "exp6_real_samples.png", title="真实 MNIST 样本")

    print_header("模型构建")
    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    print("生成器 Generator：")
    print(generator)
    print("\n判别器 Discriminator：")
    print(discriminator)
    print("说明：判别器 forward 返回 logit，torch.sigmoid(logit) 表示输入图像为真实图片的概率。")

    history, saved_samples, _ = train_gan(
        train_loader=train_loader,
        generator=generator,
        discriminator=discriminator,
        device=device,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        lr=args.lr,
        sample_every=args.sample_every,
        output_dir=OUTPUT_DIR,
    )

    print_header("结果可视化")
    plot_loss_curve(history, OUTPUT_DIR / "exp6_loss_curve.png")
    plot_score_curve(history, OUTPUT_DIR / "exp6_discriminator_score_curve.png")
    save_generation_progress(saved_samples, OUTPUT_DIR / "exp6_generation_progress.png")
    save_latent_interpolation(
        generator,
        latent_dim=args.latent_dim,
        device=device,
        output_path=OUTPUT_DIR / "exp6_latent_interpolation.png",
    )

    print_header("保存输出")
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(OUTPUT_DIR / "exp6_training_history.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "exp6_training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "model_state_dict": generator.state_dict(),
            "latent_dim": args.latent_dim,
            "random_seed": RANDOM_SEED,
        },
        OUTPUT_DIR / "exp6_generator.pt",
    )
    torch.save(
        {
            "model_state_dict": discriminator.state_dict(),
            "random_seed": RANDOM_SEED,
        },
        OUTPUT_DIR / "exp6_discriminator.pt",
    )
    write_summary(
        OUTPUT_DIR / "exp6_metrics_summary.txt",
        args=args,
        data_source=data_source,
        device=device,
        train_size=len(train_dataset),
        history=history,
    )

    print_header("已保存文件列表")
    for file_path in sorted(OUTPUT_DIR.glob("*")):
        if file_path.is_file():
            print(file_path)


if __name__ == "__main__":
    main()
