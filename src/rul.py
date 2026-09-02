# -*- coding: utf-8 -*-
"""
刀具剩余寿命（RUL）预测 —— 寿命预测模块。

任务：对 AI4I 2020 预处理数据回归预测 RUL（剩余可磨损时长）。
模型：
  - 随机森林 RandomForestRegressor（基线）
  - 梯度提升 GradientBoostingRegressor（主模型）
  - MLPRegressor（神经网络备选，特征已 Z-score 标准化）
超参数：遗传算法（src.ga.GeneticOptimizer）寻优（RF / GBR），目标为最小化验证集 MAE。
评估：MAE / RMSE / R²（test 集）。

运行（在项目根目录 d:/ljy 下）：
    python -m src.rul                       # 梯度提升 + GA 寻优
    python -m src.rul --model rf --no-ga    # 随机森林基线
    python -m src.rul --model mlp --no-ga   # MLP 神经网络
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor

from . import paths
from .ga import GeneticOptimizer

# AI4I 预处理数据中的元数据/目标列（其余均为特征）。
# 关键：RUL 由 Tool wear [min] 直接构造（RUL = max(0, 253 - wear)），
# 因此磨损量本身（标准化的 Tool wear [min] 与原始 wear_raw）必须从特征中剔除，
# 否则模型只是对磨损量做反变换，属于数据泄漏，R² 会虚假地接近 1。
# 真实任务：仅由加工过程信号（温度/扭矩/转速/类型/派生特征）推断剩余寿命。
META_COLS = ["UDI", "split", "RUL", "Machine failure", "wear_raw", "Tool wear [min]"]

# 遗传算法搜索空间（RF / GBR）
RF_SPACE = {
    "n_estimators": {"type": "int", "low": 80, "high": 300},
    "max_depth": {"type": "int", "low": 6, "high": 30},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 8},
    "max_features": {"type": "choice", "values": ["sqrt", "log2", 0.5, None]},
}
GBR_SPACE = {
    "n_estimators": {"type": "int", "low": 80, "high": 300},
    "max_depth": {"type": "int", "low": 2, "high": 7},
    "learning_rate": {"type": "float", "low": 0.02, "high": 0.25},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
}


def load_data():
    """读取 AI4I 预处理数据，返回 (df, X, y, 特征列名)。"""
    df = pd.read_csv(paths.AI4I_PROCESSED)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].astype(float)
    y = df["RUL"].astype(float)
    return df, X, y, feat_cols


def make_model(model_name, params):
    if model_name == "rf":
        return RandomForestRegressor(**params)
    if model_name == "mlp":
        return MLPRegressor(**params)
    return GradientBoostingRegressor(**params)


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main(model_name="gbr", use_ga=True, seed=42):
    df, X, y, feat_cols = load_data()
    tr = df["split"].eq("train").to_numpy()
    va = df["split"].eq("val").to_numpy()
    te = df["split"].eq("test").to_numpy()
    Xte, yte = X[te], y[te]
    # 调参/训练用 train+val，test 全程独立（仅最终评估）
    Xtrv = pd.concat([X[tr], X[va]]).reset_index(drop=True)
    ytrv = pd.concat([y[tr], y[va]]).reset_index(drop=True)
    print(f"[rul] 数据: {X.shape}，train={tr.sum()} val={va.sum()} test={te.sum()}")
    print(f"[rul] 特征数: {len(feat_cols)}（已剔除磨损量本身，避免泄漏）")

    if use_ga and model_name in ("gbr", "rf"):
        space = GBR_SPACE if model_name == "gbr" else RF_SPACE
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        print(f"[rul] 遗传算法寻优（{model_name}，适应度=3折CV neg-MAE）...")

        def fitness(ind):
            m = make_model(model_name, ind)
            m.set_params(random_state=seed)
            return cross_val_score(m, Xtrv, ytrv, cv=cv,
                                   scoring="neg_mean_absolute_error").mean()

        ga = GeneticOptimizer(space, fitness, pop_size=10, n_generations=6, seed=seed)
        best_params, _ = ga.run()
        print(f"[rul] GA 最佳参数: {best_params}")
    else:
        if model_name == "rf":
            best_params = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1}
        elif model_name == "mlp":
            best_params = {"hidden_layer_sizes": (64, 32), "max_iter": 800,
                           "early_stopping": True, "random_state": seed}
        else:
            best_params = {"n_estimators": 300, "max_depth": 4,
                           "learning_rate": 0.08, "subsample": 1.0}

    # 用最优参数在 train+val 上重训，test 评估
    model = make_model(model_name, best_params)
    if model_name != "mlp":
        model.set_params(random_state=seed)
    model.fit(Xtrv, ytrv)
    yte_pred = model.predict(Xte)
    metrics = evaluate(yte, yte_pred)
    print(f"[rul] test 指标: MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
          f"R²={metrics['r2']:.4f}")

    # 特征重要性（树模型有 feature_importances_）
    if hasattr(model, "feature_importances_"):
        imp = sorted(zip(feat_cols, model.feature_importances_),
                     key=lambda t: t[1], reverse=True)
        print("[rul] 特征重要性 Top5:")
        for name, v in imp[:5]:
            print(f"    {name}: {v:.4f}")

    # 保存模型与结果
    joblib.dump(model, os.path.join(paths.MODELS_DIR, "rul_model.joblib"))
    with open(os.path.join(paths.MODELS_DIR, "rul_best_params.json"),
              "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "params": best_params, "seed": seed,
                   "fitness": "3-fold CV neg-MAE"},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(paths.MODELS_DIR, "rul_metrics.json"),
              "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[rul] 模型已保存到 {paths.MODELS_DIR}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RUL 预测训练")
    ap.add_argument("--model", choices=["gbr", "rf", "mlp"], default="gbr")
    ap.add_argument("--no-ga", action="store_true", help="跳过遗传算法寻优，用默认参数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.model, not args.no_ga, args.seed)
