# -*- coding: utf-8 -*-
"""
样例生成：从预处理后的 test 集抽取若干样本，反向还原为「原始特征」，供前端演示
「原始输入 → 标准化 → 预测」的完整链路，并可对比真实标签验证一致性。
"""
import pandas as pd

from . import config, registry

PARAM_ORDER = ["N", "ap", "ae", "F", "Z", "D"]


def status_samples(n_per_class: int = 2, seed: int = 42):
    """返回状态识别的原始特征样例列表（每类 n_per_class 个）。"""
    st = registry.ensure_loaded()
    scaler = st["msm_scaler"]
    df = pd.read_csv(config.MSM_FEATURES)
    te = df[df["split"] == "test"]

    samples = []
    for label in (0, 1, 2):
        sub = te[te["label"] == label].sample(n=min(n_per_class, len(te[te["label"] == label])),
                                              random_state=seed)
        for _, row in sub.iterrows():
            d = str(row["direction"])
            if d not in ("UP", "DOWN"):
                d = "UP" if int(row["dir_UP"]) == 1 else "DOWN"
            # 信号特征反标准化：raw = scaled * std + mean
            signals = {name: float(row[name]) * sc["std"] + sc["mean"]
                       for name, sc in scaler.items()}
            samples.append({
                "id": f"{row['machine']}/{row['dataset']}",
                "direction": d,
                "params": {p: float(row[f"param_{p}"]) for p in PARAM_ORDER},
                "signals": signals,
                "label": int(label),
                "label_name": config.STATUS_LABELS[int(label)],
            })
    return samples


def rul_samples(n: int = 5, seed: int = 42):
    """返回 RUL 的原始过程特征样例列表。"""
    st = registry.ensure_loaded()
    scaler = st["ai4i_scaler"]
    df = pd.read_csv(config.AI4I_PROCESSED)
    te = df[df["split"] == "test"].sample(n=n, random_state=seed)

    def unscale(row, k):
        return float(row[k]) * scaler[k]["std"] + scaler[k]["mean"]

    samples = []
    for _, row in te.iterrows():
        t = "M"
        if int(row["Type_L"]) == 1:
            t = "L"
        elif int(row["Type_H"]) == 1:
            t = "H"
        samples.append({
            "id": f"样本{len(samples) + 1}",
            "type": t,
            "air_temperature": unscale(row, "Air temperature [K]"),
            "process_temperature": unscale(row, "Process temperature [K]"),
            "rotational_speed": unscale(row, "Rotational speed [rpm]"),
            "torque": unscale(row, "Torque [Nm]"),
            "rul": float(row["RUL"]),
        })
    return samples
