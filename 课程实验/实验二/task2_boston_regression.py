#!/usr/bin/env python
# coding: utf-8

# 任务2：Boston 回归
# 代码逻辑我跟 ipynb 对齐过了，py 这边就多塞一点点注释

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from importlib.resources import files
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 42


def manual_standardize(train_df, test_df):
    """
    按实验题要求手动标准化。

    只用训练集统计量，是为了避免测试集信息泄露。
    """
    mean = train_df.mean()
    std = train_df.std(ddof=0).replace(0, 1)
    train_scaled = (train_df - mean) / std
    test_scaled = (test_df - mean) / std
    return train_scaled, test_scaled, mean, std


def load_boston_compatible():
    """
    优先读取 sklearn 安装目录中的 boston_house_prices.csv。

    如果当前环境中确实读不到 Boston 数据，
    就回退到 diabetes 数据集，保证 notebook 至少能运行。
    """
    try:
        path = files("sklearn.datasets.data").joinpath("boston_house_prices.csv")
        raw = pd.read_csv(path, skiprows=1)
        X = raw.drop(columns=["MEDV"])
        y = raw["MEDV"]
        source = f"Boston Housing (sklearn 内置 CSV): {path}"
        return X, y, source
    except Exception as exc:
        from sklearn.datasets import load_diabetes

        dataset = load_diabetes(as_frame=True)
        X = dataset.data.copy()
        y = pd.Series(dataset.target, name="target")
        source = f"Boston 数据加载失败，已回退到 diabetes 数据集。原因：{exc}"
        return X, y, source


X, y, data_source = load_boston_compatible()

boston_feature_desc = {
    "CRIM": "城镇人均犯罪率",
    "ZN": "住宅用地超过 25000 平方英尺的比例",
    "INDUS": "城镇非零售商业用地比例",
    "CHAS": "是否邻近查尔斯河（1 是，0 否）",
    "NOX": "一氧化氮浓度",
    "RM": "每套住宅的平均房间数",
    "AGE": "1940 年前建成的自住房比例",
    "DIS": "到五个波士顿就业中心的加权距离",
    "RAD": "公路可达性指数",
    "TAX": "每 10000 美元的房产税率",
    "PTRATIO": "城镇师生比",
    "B": "与黑人比例相关的统计量",
    "LSTAT": "低收入人口比例",
    "MEDV": "自住房屋价格中位数（目标变量）",
}

print(data_source)
print("\n数据规模：", X.shape)
print("特征列：", list(X.columns))
print("\n前 5 行数据：")
display(X.head())
print("\n描述性统计：")
display(X.describe())

if set(X.columns).issubset(set(boston_feature_desc)):
    feature_desc_df = pd.DataFrame(
        {"Feature": list(X.columns), "Description": [boston_feature_desc[col] for col in X.columns]}
    )
    print("\n特征说明：")
    display(feature_desc_df)


# 先看一下目标值分布，不然回归那块会有点没底
plt.figure(figsize=(8, 5))
sns.histplot(y, bins=25, kde=True)
plt.title("目标变量分布")
plt.xlabel(y.name if y.name is not None else "Target")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=RANDOM_STATE,
)

X_train_scaled, X_test_scaled, train_mean, train_std = manual_standardize(X_train, X_test)

print("训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)


# 三个模型放一起比较，后面出结果会方便很多
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "RandomForestRegressor": RandomForestRegressor(random_state=RANDOM_STATE),
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred

    results.append(
        {
            "Model": name,
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
        }
    )

results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
display(results_df)


# 分开画会好读一点，不然挤一张图里有点晕
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metrics = ["MSE", "MAE", "R2"]

for ax, metric in zip(axes, metrics):
    sns.barplot(data=results_df, x="Model", y=metric, ax=ax, palette="Set2")
    ax.set_title(f"{metric} 对比")
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.show()


# 这个图挺常用的，看预测值有没有贴近真实值
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
y_min, y_max = float(np.min(y_test)), float(np.max(y_test))

for ax, (name, y_pred) in zip(axes, predictions.items()):
    ax.scatter(y_test, y_pred, alpha=0.75)
    ax.plot([y_min, y_max], [y_min, y_max], color="red", linestyle="--", linewidth=2)
    ax.set_title(name)
    ax.set_xlabel("真实值")
    ax.set_ylabel("预测值")

plt.suptitle("预测值 vs 真实值")
plt.tight_layout()
plt.show()
