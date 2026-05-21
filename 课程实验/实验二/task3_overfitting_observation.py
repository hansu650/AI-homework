#!/usr/bin/env python
# coding: utf-8

# 任务3：决策树过拟合观察
# 这个版本也按 ipynb 同步过了，主要就是多留一点点人味注释

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 42

# 这里还是用 Iris，数据小一点，做这个观察实验刚刚好
iris = load_iris(as_frame=True)
X = iris.data.copy()
y = iris.target.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y,
)

# 题目给定的深度范围，None 表示不限制
depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]


records = []

for depth in depth_values:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    # 训练集和测试集都记一下，后面画曲线就靠这个了
    records.append(
        {
            "max_depth": "None" if depth is None else depth,
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
        }
    )

score_df = pd.DataFrame(records)
display(score_df)


x_positions = list(range(1, len(score_df) + 1))
depth_labels = score_df["max_depth"].astype(str).tolist()

# 找测试集表现最好的点，基本就能当作一个比较合适的复杂度
best_idx = score_df["test_accuracy"].idxmax()
best_position = x_positions[best_idx]
best_depth = depth_labels[best_idx]
best_score = score_df.loc[best_idx, "test_accuracy"]

plt.figure(figsize=(11, 6))
plt.plot(x_positions, score_df["train_accuracy"], marker="o", linewidth=2, label="训练集准确率")
plt.plot(x_positions, score_df["test_accuracy"], marker="s", linewidth=2, label="测试集准确率")

plt.axvspan(0.5, 2.5, color="#f39c12", alpha=0.15, label="欠拟合区域（浅层树）")
plt.axvline(best_position, color="#27ae60", linestyle="--", linewidth=2, label=f"最优拟合点（depth={best_depth}）")

if best_position < x_positions[-1]:
    plt.axvspan(
        best_position + 0.5,
        x_positions[-1] + 0.5,
        color="#e74c3c",
        alpha=0.12,
        label="过拟合区域（更深的树）",
    )

# 这块标一下，不然有时候老师看图会问你最佳点是哪个
plt.annotate(
    f"best={best_score:.3f}",
    xy=(best_position, best_score),
    xytext=(best_position + 0.2, best_score - 0.08),
    arrowprops={"arrowstyle": "->", "color": "black"},
)

plt.xticks(x_positions, depth_labels)
plt.ylim(0.5, 1.05)
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("模型复杂度（树深度）与泛化性能")
plt.legend()
plt.tight_layout()
plt.show()
