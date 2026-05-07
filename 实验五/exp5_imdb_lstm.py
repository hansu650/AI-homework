#!/usr/bin/env python
# coding: utf-8

"""
实验5：IMDB 影评情感分类

这份代码按实验报告的顺序写：先读取数据，再做文本清洗、词表、
LSTM 训练、测试集评估，最后把报告里要用的图和文字材料都保存下来。
"""

import html
import json
import random
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore")

# Windows 里把输出重定向到日志时，手动设成 UTF-8，截图和保存日志都更稳。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

# 画图时尽量用中文字体，后面截图放进报告会更清楚。
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# 这些参数集中放在这里，后面想改实验设置比较方便。
RANDOM_SEED = 42
MAX_LEN = 200
MAX_VOCAB_SIZE = 20000
MIN_WORD_FREQ = 2
BATCH_SIZE = 64
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3
PATIENCE = 2
MIN_DELTA = 1e-4
EMBED_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.5
BIDIRECTIONAL = True
NUM_WORKERS = 0

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "IMDB-Dataset.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs_exp5"
BEST_MODEL_PATH = OUTPUT_DIR / "exp5_best_lstm.pt"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}


def print_header(title: str) -> None:
    print(f"\n========== {title} ==========")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """把几个随机源固定住，减少每次运行之间的波动。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """有 CUDA 就用显卡，没有就用 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_text(text: str) -> str:
    """清洗 HTML 和奇怪符号，保留英文情感词本身。"""
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """英文分词这里用正则就够了，重点是流程清楚、可复现。"""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def build_vocab(token_lists: Iterable[Sequence[str]], max_vocab_size: int, min_freq: int) -> Dict[str, int]:
    """只用训练集建词表，避免验证集和测试集信息提前泄漏。"""
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, count in counter.most_common(max_vocab_size - len(vocab)):
        if count < min_freq:
            continue
        vocab[word] = len(vocab)
    return vocab


def encode_tokens(tokens: Sequence[str], vocab: Dict[str, int], max_len: int) -> Tuple[List[int], int]:
    """把一句影评变成固定长度的数字序列，短了补 0，长了截断。"""
    ids = [vocab.get(token, UNK_IDX) for token in tokens[:max_len]]
    length = max(1, len(ids))
    if len(ids) < max_len:
        ids = ids + [PAD_IDX] * (max_len - len(ids))
    return ids, length


class IMDBReviewDataset(Dataset):
    """DataLoader 每次取出 input_ids、真实长度和标签。"""

    def __init__(self, frame: pd.DataFrame, vocab: Dict[str, int], max_len: int) -> None:
        sequences: List[List[int]] = []
        lengths: List[int] = []
        labels: List[int] = []

        for _, row in frame.iterrows():
            ids, length = encode_tokens(row["tokens"], vocab, max_len)
            sequences.append(ids)
            lengths.append(length)
            labels.append(int(row["label"]))

        self.input_ids = torch.tensor(np.array(sequences, dtype=np.int64), dtype=torch.long)
        self.lengths = torch.tensor(np.array(lengths, dtype=np.int64), dtype=torch.long)
        self.labels = torch.tensor(np.array(labels, dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.lengths[index], self.labels[index]


class LSTMSentimentClassifier(nn.Module):
    """Embedding + LSTM + Dropout + Linear，最后输出一个正类 logit。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        pad_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)

        # pack 后 LSTM 不会把 padding 当作有效单词去读。
        lengths_cpu = lengths.detach().cpu()
        packed = pack_padded_sequence(embedded, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)

        if self.bidirectional:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden = hidden[-1]

        logits = self.fc(self.dropout(final_hidden)).squeeze(1)
        return logits


def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Tuple[float, float]:
    """训练和验证共用这一段，有 optimizer 时才会反向更新。"""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for input_ids, lengths, labels in dataloader:
            input_ids = input_ids.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (preds == labels.long()).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """按 epoch 训练，同时记录 EarlyStopping 需要的信息。"""
    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    stopped_early = False

    print_header("Training Logs")
    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, device, optimizer=None)
        seconds = time.time() - start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "seconds": seconds,
        }
        history.append(row)

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
            f"time={seconds:.2f}s"
        )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                stopped_early = True
                print(f"EarlyStopping triggered at epoch {epoch}.")
                break

    info = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "patience": PATIENCE,
        "stopped_early": stopped_early,
        "trained_epochs": len(history),
    }
    return history, info


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """收集测试集真实标签、正类概率和预测类别。"""
    model.eval()
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for input_ids, lengths, labels in dataloader:
            input_ids = input_ids.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            logits = model(input_ids, lengths)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_labels.extend(labels.numpy().astype(int).tolist())
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def save_plot(path: Path) -> None:
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_label_distribution(df: pd.DataFrame, output_path: Path) -> None:
    counts = df["sentiment"].value_counts().reindex(["negative", "positive"])
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=counts.index, y=counts.values, palette=["#4C78A8", "#F58518"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    plt.title("IMDB 正负样本数量分布")
    plt.xlabel("sentiment")
    plt.ylabel("count")
    save_plot(output_path)


def plot_length_distribution(lengths: Sequence[int], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(lengths, bins=60, color="#4C78A8")
    plt.axvline(MAX_LEN, color="#E45756", linestyle="--", linewidth=2, label=f"maxlen={MAX_LEN}")
    plt.title("影评长度分布")
    plt.xlabel("token length")
    plt.ylabel("review count")
    plt.legend()
    save_plot(output_path)


def plot_training_loss(history: List[Dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train_loss")
    plt.plot(epochs, [row["val_loss"] for row in history], marker="s", label="val_loss")
    plt.title("训练集和验证集 Loss 曲线")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xticks(epochs)
    plt.legend()
    save_plot(output_path)


def plot_accuracy_curve(history: List[Dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train_acc"] for row in history], marker="o", label="train_acc")
    plt.plot(epochs, [row["val_acc"] for row in history], marker="s", label="val_acc")
    plt.title("训练集和验证集 Accuracy 曲线")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0.45, 1.0)
    plt.xticks(epochs)
    plt.legend()
    save_plot(output_path)


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"],
    )
    plt.title("测试集混淆矩阵")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    save_plot(output_path)


def plot_metrics_bar(metrics: Dict[str, float], output_path: Path) -> None:
    names = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics[name.lower()] for name in names]
    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(x=names, y=values, palette=["#4C78A8", "#59A14F", "#F58518", "#E45756"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f")
    plt.ylim(0, 1.0)
    plt.title("测试集评价指标")
    plt.ylabel("score")
    save_plot(output_path)


def predict_review(
    model: nn.Module,
    review: str,
    vocab: Dict[str, int],
    device: torch.device,
) -> Dict[str, object]:
    cleaned = clean_text(review)
    tokens = tokenize(cleaned)
    ids, length = encode_tokens(tokens, vocab, MAX_LEN)

    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([ids], dtype=torch.long).to(device)
        lengths = torch.tensor([length], dtype=torch.long).to(device)
        prob = torch.sigmoid(model(input_ids, lengths)).item()

    pred_id = 1 if prob >= 0.5 else 0
    return {
        "review": review,
        "cleaned": cleaned,
        "prob_positive": prob,
        "pred_label": ID_TO_LABEL[pred_id],
    }


def save_prediction_examples(examples: List[Dict[str, object]], output_path: Path) -> None:
    lines = ["Experiment 5 custom prediction examples", ""]
    for idx, item in enumerate(examples, start=1):
        lines.append(f"[{idx}] Review: {item['review']}")
        lines.append(f"    Predicted label: {item['pred_label']}")
        lines.append(f"    Positive probability: {item['prob_positive']:.4f}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_wrong_examples(
    test_frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    max_items: int = 10,
) -> None:
    wrong_indices = np.where(y_true != y_pred)[0][:max_items]
    lines = ["Experiment 5 wrong prediction examples", ""]
    if len(wrong_indices) == 0:
        lines.append("本次测试集中没有收集到预测错误的样本。")
    for order, idx in enumerate(wrong_indices, start=1):
        row = test_frame.iloc[int(idx)]
        review = re.sub(r"\s+", " ", str(row["review"])).strip()
        if len(review) > 500:
            review = review[:500] + "..."
        lines.append(f"[{order}] True: {ID_TO_LABEL[int(y_true[idx])]} | Pred: {ID_TO_LABEL[int(y_pred[idx])]} | Positive prob: {y_prob[idx]:.4f}")
        lines.append(f"    Review: {review}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_metrics_text(path: Path, metrics: Dict[str, float], matrix: np.ndarray, early_info: Dict[str, float]) -> None:
    """把关键指标另存成纯文本，报告截图时更直接。"""
    lines = [
        f"Accuracy : {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall   : {metrics['recall']:.4f}",
        f"F1       : {metrics['f1']:.4f}",
        "Confusion Matrix:",
        f"[[{int(matrix[0, 0])}, {int(matrix[0, 1])}],",
        f" [{int(matrix[1, 0])}, {int(matrix[1, 1])}]]",
        f"Best Epoch: {int(early_info['best_epoch'])}",
        f"Best Val Loss: {early_info['best_val_loss']:.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_header("Read Experiment Requirement")
    print("任务: 使用 IMDB-Dataset.csv 完成影评正负情感二分类。")
    print("要求: train/val/test=80%/10%/10%，maxlen=200，LSTM，EarlyStopping。")
    print("输出: Accuracy、Precision、Recall、F1、混淆矩阵、训练曲线和自定义预测。")

    device = get_device()

    print_header("Experiment 5: IMDB Sentiment Classification")
    print(f"Current device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"maxlen: {MAX_LEN}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"epochs: {NUM_EPOCHS}")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["label"] = df["sentiment"].map(LABEL_TO_ID).astype(int)

    print_header("Dataset Information")
    print(f"数据集路径: {DATA_PATH}")
    print(f"数据总量: {len(df)}")
    label_counts = df["sentiment"].value_counts().reindex(["negative", "positive"])
    print(f"negative 数量: {int(label_counts['negative'])}")
    print(f"positive 数量: {int(label_counts['positive'])}")
    plot_label_distribution(df, OUTPUT_DIR / "exp5_label_distribution.png")

    print_header("Text Preprocessing")
    df["clean_review"] = df["review"].apply(clean_text)
    df["tokens"] = df["clean_review"].apply(tokenize)
    df["review_length"] = df["tokens"].apply(len)
    print("清洗方式: 去掉 HTML 标签、转小写、去掉多数标点，再用正则分词。")
    print(f"maxlen: {MAX_LEN}")
    print(f"影评平均 token 数: {df['review_length'].mean():.2f}")
    print(f"影评长度中位数: {df['review_length'].median():.0f}")
    print(f"影评最长 token 数: {df['review_length'].max()}")
    print("清洗示例:")
    print("原文:", str(df.iloc[0]["review"])[:240].replace("\n", " "), "...")
    print("清洗后:", df.iloc[0]["clean_review"][:240], "...")
    plot_length_distribution(df["review_length"], OUTPUT_DIR / "exp5_review_length_distribution.png")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"train 数量: {len(train_df)}")
    print(f"val 数量: {len(val_df)}")
    print(f"test 数量: {len(test_df)}")

    vocab = build_vocab(train_df["tokens"], MAX_VOCAB_SIZE, MIN_WORD_FREQ)
    vocab_info = {
        "max_vocab_size": MAX_VOCAB_SIZE,
        "actual_vocab_size": len(vocab),
        "min_word_freq": MIN_WORD_FREQ,
        "pad_idx": PAD_IDX,
        "unk_idx": UNK_IDX,
    }
    write_json(OUTPUT_DIR / "exp5_vocab_info.json", vocab_info)

    print_header("Vocabulary Information")
    print(f"vocab_size: {len(vocab)}")
    print(f"vocab_size 上限: {MAX_VOCAB_SIZE}")
    print(f"min_word_freq: {MIN_WORD_FREQ}")
    print(f"PAD token/index: {PAD_TOKEN}/{PAD_IDX}")
    print(f"UNK token/index: {UNK_TOKEN}/{UNK_IDX}")
    print("词表前 20 个词:", list(vocab.keys())[:20])

    train_dataset = IMDBReviewDataset(train_df, vocab, MAX_LEN)
    val_dataset = IMDBReviewDataset(val_df, vocab, MAX_LEN)
    test_dataset = IMDBReviewDataset(test_df, vocab, MAX_LEN)
    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)
    test_loader = make_loader(test_dataset, shuffle=False)

    model = LSTMSentimentClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print_header("Model Structure")
    print(model)

    history, early_info = train_model(model, train_loader, val_loader, criterion, optimizer, device)

    print_header("EarlyStopping Information")
    print(f"patience: {early_info['patience']}")
    print(f"trained_epochs: {early_info['trained_epochs']}")
    print(f"best_epoch: {early_info['best_epoch']}")
    print(f"best_val_loss: {early_info['best_val_loss']:.4f}")
    print(f"stopped_early: {early_info['stopped_early']}")
    print(f"best_model_path: {BEST_MODEL_PATH}")

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    y_true, y_prob, y_pred = collect_predictions(model, test_loader, device)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    print_header("Test Metrics")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(matrix)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")

    plot_training_loss(history, OUTPUT_DIR / "exp5_training_loss_curve.png")
    plot_accuracy_curve(history, OUTPUT_DIR / "exp5_accuracy_curve.png")
    plot_confusion_matrix(matrix, OUTPUT_DIR / "exp5_confusion_matrix.png")
    plot_metrics_bar(metrics, OUTPUT_DIR / "exp5_metrics_bar.png")

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "exp5_training_history.csv", index=False, encoding="utf-8-sig")
    write_json(OUTPUT_DIR / "exp5_metrics.json", metrics)
    write_metrics_text(OUTPUT_DIR / "exp5_metrics.txt", metrics, matrix, early_info)
    write_json(
        OUTPUT_DIR / "exp5_split_info.json",
        {
            "total": len(df),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "negative": int(label_counts["negative"]),
            "positive": int(label_counts["positive"]),
            "maxlen": MAX_LEN,
            "vocab_size": len(vocab),
        },
    )

    sample_reviews = [
        "This movie is fantastic!",
        "A waste of time.",
        "The story is boring and too long.",
        "I really like the actors and the ending is touching.",
        "The film starts slowly, but the ending is powerful and very emotional.",
        "I expected much more from this movie, but it was messy, dull and too long.",
    ]
    sample_predictions = [predict_review(model, review, vocab, device) for review in sample_reviews]
    save_prediction_examples(sample_predictions, OUTPUT_DIR / "exp5_prediction_examples.txt")
    save_wrong_examples(test_df, y_true, y_prob, y_pred, OUTPUT_DIR / "exp5_wrong_examples.txt")

    print_header("Sample Predictions")
    for item in sample_predictions:
        print(f"Review: {item['review']}")
        print(f"Predicted label: {item['pred_label']} | Positive probability: {item['prob_positive']:.4f}")

    print_header("Saved Output Files")
    files_to_show = [
        Path(__file__),
        BEST_MODEL_PATH,
        OUTPUT_DIR / "exp5_label_distribution.png",
        OUTPUT_DIR / "exp5_review_length_distribution.png",
        OUTPUT_DIR / "exp5_training_loss_curve.png",
        OUTPUT_DIR / "exp5_accuracy_curve.png",
        OUTPUT_DIR / "exp5_confusion_matrix.png",
        OUTPUT_DIR / "exp5_metrics_bar.png",
        OUTPUT_DIR / "exp5_prediction_examples.txt",
        OUTPUT_DIR / "exp5_wrong_examples.txt",
        OUTPUT_DIR / "exp5_training_history.csv",
        OUTPUT_DIR / "exp5_metrics.json",
        OUTPUT_DIR / "exp5_metrics.txt",
        OUTPUT_DIR / "exp5_split_info.json",
        OUTPUT_DIR / "exp5_vocab_info.json",
    ]
    for file_path in files_to_show:
        print(file_path)


if __name__ == "__main__":
    main()
