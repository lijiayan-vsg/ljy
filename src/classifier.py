# -*- coding: utf-8 -*-
"""
刀具状态识别（三分类）—— 模式与图像识别模块。

任务：对 MSM 多传感器特征矩阵做「正常(0) / 加工异常(1) / 刀具缺陷(2)」三分类。
模型：随机森林 RandomForestClassifier（主）+ 梯度提升 GradientBoostingClassifier（备选），
      类别不平衡用 class_weight='balanced' 处理。
超参数：遗传算法（src.ga.GeneticOptimizer）寻优，适应度 = 3 折交叉验证 macro-F1
      （比单一验证集更稳健，避免对某一切分过拟合）。
评估：accuracy / macro-F1 / weighted-F1 / 混淆矩阵 / 分类报告（test 集，全程不参与调参）。

运行（在项目根目录 d:/ljy 下）：
    python -m src.classifier                 # 随机森林 + GA 寻优
    python -m src.classifier --no-ga         # 跳过 GA，快速基线
    python -m src.classifier --model gb      # 梯度提升 + GA 寻优
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import paths
from .ga import GeneticOptimizer

# MSM 特征矩阵中的元数据列（其余均为特征）
META_COLS = ["machine", "dataset", "label", "split", "direction"]

# 遗传算法搜索空间（max_depth 允许 None = 树生长到叶节点纯/样本数下限）
RF_SPACE = {
    "n_estimators": {"type": "int", "low": 80, "high": 260},
    "max_depth": {"type": "choice", "values": [None, 6, 8, 10, 12, 14, 16, 18, 20, 24]},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 8},
    "max_features": {"type": "choice", "values": ["sqrt", "log2", None]},
}
GB_SPACE = {
    "n_estimators": {"type": "int", "low": 80, "high": 260},
    "max_depth": {"type": "int", "low": 2, "high": 7},
    "learning_rate": {"type": "float", "low": 0.03, "high": 0.3},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
}


def load_data():
    """读取 MSM 特征矩阵，返回 (df, X, y, 特征列名)。"""
    df = pd.read_csv(paths.MSM_FEATURES)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].astype(float)
    y = df["label"].astype(int)
    return df, X, y, feat_cols


def make_model(model_name, params):
    if model_name == "gb":
        return GradientBoostingClassifier(**params)
    return RandomForestClassifier(**params)


def evaluate(y_true, y_pred):
    """在 test 集上计算分类指标。"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=3),
    }


def main(model_name="rf", use_ga=True, seed=42):
    df, X, y, feat_cols = load_data()
    tr = df["split"].eq("train").to_numpy()
    va = df["split"].eq("val").to_numpy()
    te = df["split"].eq("test").to_numpy()
    Xte, yte = X[te], y[te]
    # 调参/训练用 train+val，test 全程独立（仅最终评估）
    Xtrv = pd.concat([X[tr], X[va]]).reset_index(drop=True)
    ytrv = pd.concat([y[tr], y[va]]).reset_index(drop=True)
    print(f"[classifier] 数据: {X.shape}，train={tr.sum()} val={va.sum()} test={te.sum()}")
    print(f"[classifier] 特征数: {len(feat_cols)}，训练集(t+va)类别分布: "
          f"{np.bincount(ytrv, minlength=3).tolist()}")

    space = GB_SPACE if model_name == "gb" else RF_SPACE
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    if use_ga:
        print(f"[classifier] 遗传算法寻优（{model_name}，适应度=3折CV macro-F1）...")

        def fitness(ind):
            m = make_model(model_name, ind)
            m.set_params(random_state=seed, class_weight="balanced")
            return cross_val_score(m, Xtrv, ytrv, cv=cv, scoring="f1_macro").mean()

        ga = GeneticOptimizer(space, fitness, pop_size=10, n_generations=6, seed=seed)
        best_params, _ = ga.run()
        print(f"[classifier] GA 最佳参数: {best_params}")
    else:
        best_params = (
            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1,
             "max_features": "sqrt"}
            if model_name == "rf" else
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "subsample": 1.0}
        )

    # 用最优参数在 train+val 上重训，test 评估
    model = make_model(model_name, best_params)
    model.set_params(random_state=seed, class_weight="balanced")
    model.fit(Xtrv, ytrv)
    yte_pred = model.predict(Xte)
    metrics = evaluate(yte, yte_pred)
    print(f"[classifier] test 指标: acc={metrics['accuracy']:.4f} "
          f"macro_f1={metrics['macro_f1']:.4f}")
    print(metrics["classification_report"])

    # 保存模型与结果
    joblib.dump(model, os.path.join(paths.MODELS_DIR, "status_classifier.joblib"))
    with open(os.path.join(paths.MODELS_DIR, "status_best_params.json"),
              "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "params": best_params, "seed": seed,
                   "fitness": "3-fold CV macro-F1"},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(paths.MODELS_DIR, "status_metrics.json"),
              "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[classifier] 模型已保存到 {paths.MODELS_DIR}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="刀具状态三分类训练")
    ap.add_argument("--model", choices=["rf", "gb"], default="rf")
    ap.add_argument("--no-ga", action="store_true", help="跳过遗传算法寻优，用默认参数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.model, not args.no_ga, args.seed)
