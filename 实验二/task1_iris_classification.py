#!/usr/bin/env python
# coding: utf-8

# 任务1：Iris 分类
# 这个 py 我按 ipynb 里的代码同步过了，方便你直接复制交

# 忽略一些小警告，不然输出看着有点乱
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 先把图的风格设一下，默认的其实也能看，就是稍微朴素了点
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 42


def manual_standardize(train_df, test_df):
    """
    按实验题要求手动完成标准化：
    X_scaled = (X - mean) / std

    注意：
    1. 均值和标准差只能用训练集计算；
    2. 测试集必须使用训练集的均值和标准差来变换，
       这样才符合机器学习的规范流程。
    """
    mean = train_df.mean()
    std = train_df.std(ddof=0).replace(0, 1)
    train_scaled = (train_df - mean) / std
    test_scaled = (test_df - mean) / std
    return train_scaled, test_scaled, mean, std


# 先把数据集读进来，as_frame=True 后面处理起来会顺手很多
iris = load_iris(as_frame=True)
X = iris.data.copy()
y = iris.target.copy()
feature_names = iris.feature_names
class_names = iris.target_names

df = X.copy()
df["species"] = y.map(dict(enumerate(class_names)))

print("数据规模：", X.shape)
print("特征名称：", feature_names)
print("\n类别分布：")
display(df["species"].value_counts().rename("count").to_frame())
print("\n前 5 行数据：")
display(df.head())
print("\n描述性统计：")
display(X.describe())


# 这张图主要是看特征分布和类别能不能大致分开
sns.pairplot(
    df,
    hue="species",
    corner=True,
    diag_kind="hist",
    plot_kws={"alpha": 0.7, "s": 50},
)
plt.suptitle("Iris 特征散点矩阵", y=1.02)
plt.show()


# 按 7:3 划分，分层抽样一下更稳妥一点
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train_scaled, X_test_scaled, train_mean, train_std = manual_standardize(X_train, X_test)

print("训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)
print("\n训练集均值：")
display(train_mean.to_frame(name="mean").T)
print("\n训练集标准差：")
display(train_std.to_frame(name="std").T)


# 题目要求的 4 个模型，先都放字典里，一会循环起来比较快
models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "SVC(RBF)": SVC(),
}

results = []
fitted_models = {}

for name, model in models.items():
    # 统一用标准化后的数据训练，省得前后不一致
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    fitted_models[name] = model

    # 这里记的是题目要交的几个指标
    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }
    )

results_df = pd.DataFrame(results).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
display(results_df)


# 画成柱状图会直观点，表格有时候看着有点干
metric_columns = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
plot_df = results_df.set_index("Model")[metric_columns]

ax = plot_df.plot(kind="bar", figsize=(11, 6))
ax.set_title("四种分类模型指标对比")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)

plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# 单独训一棵浅一点的树，这样画出来不会太夸张
shallow_tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
shallow_tree.fit(X_train, y_train)

plt.figure(figsize=(16, 8))
plot_tree(
    shallow_tree,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("深度为 3 的决策树")
plt.show()
