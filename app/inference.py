# -*- coding: utf-8 -*-
"""
推理逻辑：把「原始特征」还原为模型训练时的「标准化特征向量」，再调用模型预测。

关键点（与 data/preprocess_msm.py / preprocess.py 保持一致）：
  - 状态分类：方向 one-hot + 6 个切削参数透传（不标准化）+ 229 个信号特征做 Z-score；
  - RUL：由 Type + 4 个过程量计算派生特征（温差/功率代理），再对 6 个连续量做 Z-score，
    磨损量本身不参与（避免泄漏）。
特征顺序统一取自模型自带的 feature_names_in_，避免手工对齐出错。
"""
import pandas as pd

from . import config, registry

PARAM_ORDER = ["N", "ap", "ae", "F", "Z", "D"]

# RUL 需要标准化的 6 个连续量（不含 Tool wear，防泄漏）
RUL_SCALE_KEYS = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]",
    "Temp_diff_K", "Power_proxy",
]


def _scale(signals: dict, scaler: dict) -> dict:
    """按 scaler 对信号特征做 (x-mean)/std，缺失即报错。"""
    out = {}
    for name, sc in scaler.items():
        if name not in signals:
            raise ValueError(f"缺少信号特征: {name}")
        out[name] = (float(signals[name]) - sc["mean"]) / sc["std"]
    return out


def predict_status(direction: str, params: dict, signals: dict):
    """刀具状态三分类。返回 (label:int, label_name:str, proba_dict, confidence:float)。"""
    st = registry.ensure_loaded()
    scaler = st["msm_scaler"]
    clf = st["clf"]

    feat = {}
    d = (direction or "").upper()
    feat["dir_UP"] = 1.0 if d == "UP" else 0.0
    feat["dir_DOWN"] = 1.0 if d == "DOWN" else 0.0
    for p in PARAM_ORDER:
        if p not in params:
            raise ValueError(f"缺少切削参数: {p}")
        feat[f"param_{p}"] = float(params[p])
    feat.update(_scale(signals, scaler))

    # 严格按训练特征顺序构造向量（DataFrame 带列名，sklearn 可校验列序一致）
    fn = list(clf.feature_names_in_)
    missing = [n for n in fn if n not in feat]
    if missing:
        raise ValueError(f"特征不完整，缺少: {missing[:5]}")
    x = pd.DataFrame([[feat[n] for n in fn]], columns=fn)

    label = int(clf.predict(x)[0])
    proba = clf.predict_proba(x)[0]
    classes = [int(c) for c in clf.classes_]
    proba_dict = {config.STATUS_LABELS[c]: float(proba[i]) for i, c in enumerate(classes)}
    return label, config.STATUS_LABELS[label], proba_dict, float(proba[classes.index(label)])


def predict_rul(type_: str, air: float, proc: float, speed: float, torque: float) -> float:
    """RUL 预测。输入原始过程量，返回剩余寿命（min）。"""
    st = registry.ensure_loaded()
    scaler = st["ai4i_scaler"]
    reg = st["reg"]

    t = (type_ or "M").upper()
    feat = {
        "Type_L": 1.0 if t == "L" else 0.0,
        "Type_M": 1.0 if t == "M" else 0.0,
        "Type_H": 1.0 if t == "H" else 0.0,
    }
    raw = {
        "Air temperature [K]": air,
        "Process temperature [K]": proc,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Temp_diff_K": proc - air,
        "Power_proxy": speed * torque,
    }
    for k in RUL_SCALE_KEYS:
        feat[k] = (raw[k] - scaler[k]["mean"]) / scaler[k]["std"]

    fn = list(reg.feature_names_in_)
    x = pd.DataFrame([[feat[n] for n in fn]], columns=fn)
    return max(0.0, float(reg.predict(x)[0]))
