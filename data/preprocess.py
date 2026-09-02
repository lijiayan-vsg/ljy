# -*- coding: utf-8 -*-
"""
AI4I 2020 预测性维护数据集 —— 剩余寿命（RUL）回归预处理（阶段3：数据预处理）

数据来源：UCI Machine Learning Repository（ID 601）
原始文件：data/raw/ai4i2020.csv（10,000 行 × 14 列）

预处理流程（对应 data/README.md）：
  1. 缺失值填充：数值列中位数 / 类别列众数（本数据集无缺失，保留通用逻辑）；
  2. 异常值处理：IQR（1.5 倍四分位距）截断；
  3. 特征工程：构造温差 Temp_diff_K、功率代理量 Power_proxy 等派生特征；
  4. 标签构造：由 Tool wear [min] 构造 RUL = max(0, wear_limit - wear)；
  5. 编码与归一化：Type one-hot 编码、连续列 Z-score 标准化（StandardScaler）；
  6. 输出：data/processed/ai4i_processed.csv + ai4i_scaler.json + index.json 索引。

运行：
    python data/preprocess.py [--wear-limit 253] [--seed 42]
"""
import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(DATA_DIR, "raw", "ai4i2020.csv")
PROCESSED = os.path.join(DATA_DIR, "processed")
OUT_CSV = os.path.join(PROCESSED, "ai4i_processed.csv")
OUT_SCALER = os.path.join(PROCESSED, "ai4i_scaler.json")
INDEX = os.path.join(PROCESSED, "index.json")

WEAR_LIMIT = 253.0  # 磨损上限（min），取观测最大值；RUL = max(0, limit - wear)

# 连续数值列（用于 IQR 截断 + 标准化）
NUM_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load() -> pd.DataFrame:
    return pd.read_csv(RAW)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """缺失值填充：数值列中位数、类别列众数（本数据集无缺失，逻辑保留）。"""
    for c in df.columns:
        if df[c].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode()[0])
    return df


def clip_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """IQR 1.5 倍截断：超出 [Q1-1.5IQR, Q3+1.5IQR] 的值截断到边界。"""
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        df[c] = df[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """派生特征：温差、功率代理量（转速 × 扭矩）。"""
    df["Temp_diff_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power_proxy"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"]
    return df


def build_target(df: pd.DataFrame, wear_limit: float) -> pd.DataFrame:
    """RUL = 剩余可磨损时长 = max(0, wear_limit - 当前磨损)。"""
    df["RUL"] = (wear_limit - df["Tool wear [min]"]).clip(lower=0.0)
    return df


def split_stratified(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """按 Machine failure 分层划分 train/val/test = 70/15/15。"""
    rng = np.random.default_rng(seed)
    df["split"] = "train"
    for fail in (0, 1):
        idx = df.index[df["Machine failure"] == fail].to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * 0.15)))
        n_val = max(1, int(round(n * 0.15)))
        df.loc[idx[:n_test], "split"] = "test"
        df.loc[idx[n_test:n_test + n_val], "split"] = "val"
    return df


def main(wear_limit: float = WEAR_LIMIT, seed: int = 42) -> None:
    os.makedirs(PROCESSED, exist_ok=True)
    df = load()
    print(f"[ai4i] 原始数据: {df.shape}")

    df = fill_missing(df)
    df = clip_iqr(df, NUM_COLS)
    df = feature_engineering(df)
    df = build_target(df, wear_limit)

    # 派生特征也纳入标准化
    derived = ["Temp_diff_K", "Power_proxy"]
    scale_cols = NUM_COLS + derived

    # Type one-hot（保证三列顺序稳定）
    type_dummies = pd.get_dummies(df["Type"], prefix="Type").astype(int)
    for d in ["Type_L", "Type_M", "Type_H"]:
        if d not in type_dummies:
            type_dummies[d] = 0
    type_dummies = type_dummies[["Type_L", "Type_M", "Type_H"]]

    # Z-score 标准化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df[scale_cols]), columns=scale_cols)

    out = pd.concat(
        [
            df[["UDI"]].reset_index(drop=True),
            type_dummies.reset_index(drop=True),
            X_scaled.reset_index(drop=True),
            df[["Tool wear [min]", "RUL", "Machine failure"]]
            .rename(columns={"Tool wear [min]": "wear_raw"})
            .reset_index(drop=True),
        ],
        axis=1,
    )
    out = split_stratified(out, seed)
    out.to_csv(OUT_CSV, index=False)
    print(f"[ai4i] 输出: {OUT_CSV}  shape={out.shape}")
    print(f"[ai4i] RUL 描述统计:\n{out['RUL'].describe().round(3).to_string()}")
    print(f"[ai4i] 划分: {out['split'].value_counts().to_dict()}")

    # 保存 scaler 参数（推理复用）
    scaler_json = {
        c: {"mean": float(scaler.mean_[i]), "std": float(scaler.scale_[i])}
        for i, c in enumerate(scale_cols)
    }
    with open(OUT_SCALER, "w", encoding="utf-8") as f:
        json.dump(scaler_json, f, ensure_ascii=False, indent=2)
    print(f"[ai4i] scaler 参数: {OUT_SCALER}")

    update_index(wear_limit, scale_cols, int(df["Machine failure"].sum()))


def update_index(wear_limit: float, scale_cols: list, n_fail: int) -> None:
    entry = {
        "dataset": "AI4I 2020 Predictive Maintenance",
        "role": "剩余寿命（RUL）预测（辅助数据集）",
        "source": "UCI Machine Learning Repository (ID 601)",
        "license": "CC BY 4.0",
        "raw": "data/raw/ai4i2020.csv",
        "processed": ["data/processed/ai4i_processed.csv", "data/processed/ai4i_scaler.json"],
        "samples": 10000,
        "features": scale_cols + ["Type_L", "Type_M", "Type_H"],
        "target": "RUL = max(0, wear_limit - Tool wear [min])",
        "wear_limit": wear_limit,
        "failure_samples": n_fail,
        "split": "train/val/test = 70/15/15（按 Machine failure 分层）",
        "steps": [
            "缺失值填充（数值中位数/类别众数）",
            "IQR 1.5 倍截断",
            "派生特征（温差/功率代理量）",
            "RUL 标签构造",
            "Type one-hot 编码",
            "Z-score 标准化（StandardScaler）",
            "按 Machine failure 分层划分",
        ],
    }
    _merge_index("ai4i", entry)


def _merge_index(key: str, entry: dict) -> None:
    data = {}
    if os.path.exists(INDEX):
        with open(INDEX, encoding="utf-8") as f:
            data = json.load(f)
    data.setdefault("datasets", {})
    data["datasets"][key] = entry
    data["updated"] = datetime.date.today().isoformat()
    with open(INDEX, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[ai4i] 索引更新: {INDEX}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI4I 2020 RUL 预处理")
    ap.add_argument("--wear-limit", type=float, default=WEAR_LIMIT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.wear_limit, args.seed)
