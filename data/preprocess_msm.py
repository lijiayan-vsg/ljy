# -*- coding: utf-8 -*-
"""
MSM（Multi-Sensor and MTConnect）金属铣削异常数据集 —— 刀具状态识别预处理（阶段3：数据预处理）

数据来源：Kaggle（CC BY 4.0），论文 Scientific Data 2026。
原始数据：MSM 数据集 zip（默认 D:/新建文件夹/archive_6.zip，14.2 GB）。

本脚本仅读取 zip 内的 mtc.csv（~6.4 Hz 机床控制器数据）+ label.csv（分段标注），
合计约 100 MB，无需解压 40 GB 高频信号（acc.csv / wav / mp4）。

预处理流程（对应 data/README.md）：
  1. label 解析：解析 label.csv 得到切片段 (start, end, label) 及切削参数；
  2. 信号对齐：将 mtc.csv 按时间切片到各标注段；
  3. 特征提取：时域特征（均值/标准差/RMS/最小/最大/峭度/斜率）+ 频域特征（FFT 频带能量）；
  4. 类别不平衡处理：可选 --smote（仅对训练集，需 imblearn，默认关闭）；
  5. 归一化：Z-score 标准化（StandardScaler，在训练集上拟合）；
  6. 特征矩阵：输出 msm_features.csv + msm_scaler.json；
  7. 数据划分：默认随机分层 70/15/15（三类均进 test）；--split-by-machine 为跨机泛化 held-out。

运行：
    python data/preprocess_msm.py [--zip <MSM zip 路径>] [--smote] [--split-by-machine]
"""
import argparse
import datetime
import json
import os
import zipfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(DATA_DIR, "processed")
OUT_CSV = os.path.join(PROCESSED, "msm_features.csv")
OUT_SCALER = os.path.join(PROCESSED, "msm_scaler.json")
INDEX = os.path.join(PROCESSED, "index.json")

DEFAULT_ZIP = r"D:/新建文件夹/archive_6.zip"

# 关键连续信号列（用于特征提取）
SIGNAL_COLS = [
    "S_Axis_Load", "Spindle_Speed", "X_Axis_Load", "X_Position",
    "Y_Axis_Load", "Y_Position", "Z_Axis_Load", "Z_Position",
    "energy", "watt", "Feed_Rate",
    "smapeak", "smarms", "smcrest", "smdisp", "smvrms",
    "xmapeak", "xmarms", "xmcrest", "xmdisp", "xmvrms",
    "ymapeak", "ymarms", "ymcrest", "ymdisp", "ymvrms",
    "zmapeak", "zmarms", "zmcrest", "zmdisp", "zmvrms",
]

# 频域特征列（对振动 RMS 列做 FFT 频带能量）
FFT_COLS = ["smarms", "xmarms", "ymarms", "zmarms"]

# 按机床划分：train=实验室 Hurco imi_vm20i，val=实验室 imi_vmx30ui，test=工业 Haas tmf_vf10
MACHINE_SPLIT = {
    "imi_vm20i": "train",
    "imi_vmx30ui": "val",
    "tmf_vf10": "test",
}

PARAM_COLS = ["N", "ap", "ae", "F", "Z", "D"]


def segment_features(sig: pd.DataFrame) -> dict:
    """对单个切片段的多列信号提取时域特征（7 种）+ 频域特征（3 频带）。"""
    feat = {}
    for c in SIGNAL_COLS:
        if c not in sig.columns:
            continue
        s = pd.to_numeric(sig[c], errors="coerce").dropna().to_numpy(dtype=float)
        n = s.size
        if n < 2:
            # 过短片段：所有统计量置 NaN，交由后续填充
            for stat in ["mean", "std", "rms", "min", "max", "kurtosis", "slope"]:
                feat[f"{c}__{stat}"] = np.nan
            continue
        feat[f"{c}__mean"] = float(np.mean(s))
        feat[f"{c}__std"] = float(np.std(s))
        feat[f"{c}__rms"] = float(np.sqrt(np.mean(s ** 2)))
        feat[f"{c}__min"] = float(np.min(s))
        feat[f"{c}__max"] = float(np.max(s))
        feat[f"{c}__kurtosis"] = float(pd.Series(s).kurt())
        feat[f"{c}__slope"] = float(np.polyfit(np.arange(n), s, 1)[0])
    # 频域：FFT 频带能量（均分 3 频带）
    for c in FFT_COLS:
        if c not in sig.columns:
            continue
        s = pd.to_numeric(sig[c], errors="coerce").dropna().to_numpy(dtype=float)
        if s.size < 4:
            continue
        spec = np.abs(np.fft.rfft(s)) ** 2
        total = spec.sum() + 1e-12
        for i, band in enumerate(np.array_split(spec, 3)):
            feat[f"{c}__fft_band{i + 1}"] = float(band.sum() / total)
    return feat


def process_dataset(z: zipfile.ZipFile, mtc_name: str, lab_name: str):
    """处理一个 dataset 目录：读 label.csv + mtc.csv，逐段切片并提取特征。

    返回 (特征 DataFrame, 跳过段落列表)。跳过的段落为 start/end 落在 mtc.csv 采样
    间隙内（该窗口无任何采样点）的标注段，无法从 mtc.csv 提取特征。
    """
    lab = pd.read_csv(z.open(lab_name))
    mtc = pd.read_csv(z.open(mtc_name)).sort_values("time").reset_index(drop=True)
    machine = lab_name.split("/")[0]
    dataset = lab_name.split("/")[1]
    rows = []
    skipped = []
    for _, seg in lab.iterrows():
        mask = (mtc["time"] >= seg["start"]) & (mtc["time"] <= seg["end"])
        sig = mtc.loc[mask]
        if len(sig) < 2:
            skipped.append({
                "machine": machine,
                "dataset": dataset,
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "label": int(seg["label"]),
                "n_points": int(len(sig)),
            })
            continue  # 切片段内采样点不足（落在记录间隙），跳过
        feat = segment_features(sig)
        feat["machine"] = machine
        feat["dataset"] = dataset
        feat["label"] = int(seg["label"])
        feat["direction"] = str(seg.get("direction", ""))
        for p in PARAM_COLS:
            feat[f"param_{p}"] = seg.get(p, np.nan)
        feat["tool"] = str(seg.get("Tool", ""))
        feat["workpiece"] = str(seg.get("Workpiece", ""))
        rows.append(feat)
    return pd.DataFrame(rows), skipped


def stratified_split(labels: pd.Series, seed: int) -> pd.Series:
    """按 label 分层随机划分 train/val/test = 70/15/15，保证各类别均进入 test。"""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=labels.index)
    for label in labels.unique():
        idx = labels.index[labels == label].to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * 0.15)))
        n_val = max(1, int(round(n * 0.15)))
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "val"
    return split


def main(zip_path: str, use_smote: bool, split_mode: str = "random", seed: int = 42) -> None:
    os.makedirs(PROCESSED, exist_ok=True)
    z = zipfile.ZipFile(zip_path)
    names = z.namelist()
    lab_files = [n for n in names if n.endswith("label.csv")]
    mtc_map = {n.rsplit("/", 1)[0]: n for n in names if n.endswith("mtc.csv")}

    frames = []
    all_skipped = []
    for lab_name in lab_files:
        d = lab_name.rsplit("/", 1)[0]
        if d not in mtc_map:
            continue
        fr, sk = process_dataset(z, mtc_map[d], lab_name)
        frames.append(fr)
        all_skipped.extend(sk)

    df = pd.concat(frames, ignore_index=True)
    print(f"[msm] 提取段落: {len(df)}（label.csv 标注 2575 段）")
    label_dist = df["label"].value_counts().sort_index().to_dict()
    print(f"[msm] 标签分布: {label_dist}")
    if all_skipped:
        print(f"[msm] 警告：{len(all_skipped)} 段落在 mtc.csv 采样间隙内（无采样点）被跳过：")
        for s in all_skipped:
            print(f"    {s['machine']}/{s['dataset']} label={s['label']} [{s['start']:.3f},{s['end']:.3f}] 点数={s['n_points']}")

    meta_cols = ["machine", "dataset", "label", "direction", "tool", "workpiece"]
    feat_cols = [c for c in df.columns if c not in meta_cols and not c.startswith("param_")]

    # 参数列转数值
    param_cols = [f"param_{p}" for p in PARAM_COLS]
    for c in param_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 方向 one-hot（保证两列顺序稳定）
    dir_dummy = pd.get_dummies(df["direction"], prefix="dir").astype(int)
    for d in ["dir_UP", "dir_DOWN"]:
        if d not in dir_dummy:
            dir_dummy[d] = 0
    dir_dummy = dir_dummy[["dir_UP", "dir_DOWN"]]

    # 特征矩阵：剔除近常数（低方差）列，避免标准化除零
    X = df[feat_cols].astype(float)
    keep = [c for c in X.columns if X[c].std() > 1e-9]
    dropped = [c for c in X.columns if c not in keep]
    X = X[keep].fillna(0.0)
    print(f"[msm] 特征列: {len(keep)} 个（丢弃近常数 {len(dropped)} 个）")

    # 数据划分：默认随机分层（保证三类进 test）；--split-by-machine 为跨机泛化 held-out
    if split_mode == "by_machine":
        df["split"] = df["machine"].map(MACHINE_SPLIT)
        split_desc = "按机床 train=imi_vm20i / val=imi_vmx30ui / test=tmf_vf10（工业机 held-out，跨机泛化）"
    else:
        df["split"] = stratified_split(df["label"], seed)
        split_desc = f"随机分层 train/val/test=70/15/15（按 label 分层，seed={seed}）"
    tr_mask = df["split"].eq("train")
    per_split = df.groupby("split")["label"].value_counts().unstack(fill_value=0)
    print(f"[msm] 划分方式: {split_desc}")
    print(f"[msm] 各划分标签分布:\n{per_split.to_string()}")

    # Z-score 标准化（训练集拟合，全部变换，避免数据泄漏）
    scaler = StandardScaler().fit(X.loc[tr_mask])
    X_scaled = pd.DataFrame(scaler.transform(X), columns=keep)

    # 类别不平衡：可选 SMOTE（仅训练集）
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            X_tr, y_tr = X.loc[tr_mask], df.loc[tr_mask, "label"]
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_tr, y_tr)
            bal = pd.DataFrame(X_res, columns=keep)
            bal["label"] = y_res.values
            bal["split"] = "train"
            bal.to_csv(os.path.join(PROCESSED, "msm_train_smote.csv"), index=False)
            print(f"[msm] SMOTE 后训练集: {pd.Series(y_res).value_counts().to_dict()} → msm_train_smote.csv")
        except ImportError:
            print("[msm] 未安装 imblearn，跳过 SMOTE（pip install imblearn，或训练时用 class_weight）")

    out = pd.concat(
        [
            df[["machine", "dataset", "label", "split", "direction"]].reset_index(drop=True),
            dir_dummy.reset_index(drop=True),
            df[param_cols].reset_index(drop=True),
            X_scaled.reset_index(drop=True),
        ],
        axis=1,
    )
    out.to_csv(OUT_CSV, index=False)
    print(f"[msm] 输出: {OUT_CSV}  shape={out.shape}")
    print(f"[msm] 划分: {out['split'].value_counts().to_dict()}")

    scaler_json = {
        c: {"mean": float(scaler.mean_[i]), "std": float(scaler.scale_[i])}
        for i, c in enumerate(keep)
    }
    with open(OUT_SCALER, "w", encoding="utf-8") as f:
        json.dump(scaler_json, f, ensure_ascii=False, indent=2)
    print(f"[msm] scaler 参数: {OUT_SCALER}")

    update_index(zip_path, len(df), keep, param_cols, use_smote, label_dist, len(all_skipped), per_split, split_desc)


def update_index(zip_path, n_segments, feat_cols, param_cols, use_smote, label_dist, n_skipped, per_split, split_desc):
    entry = {
        "dataset": "MSM (Multi-Sensor and MTConnect) metal cutting anomaly",
        "role": "刀具状态识别（正常/异常/刀具缺陷 三分类）+ 智能诊断（主数据集）",
        "source": "Kaggle: manufuturetoday/multi-sensor-for-metal-milling-anomaly",
        "license": "CC BY 4.0",
        "raw": zip_path,
        "processed": ["data/processed/msm_features.csv", "data/processed/msm_scaler.json"],
        "segments": n_segments,
        "label_distribution": label_dist,
        "skipped_in_gaps": n_skipped,
        "skipped_note": f"label.csv 标注 2575 段，其中 {n_skipped} 段 start/end 落在 mtc.csv 采样间隙内（该窗口无采样点）无法提取特征，已排除",
        "per_split_label_distribution": per_split.to_dict(),
        "feature_count": len(feat_cols) + len(param_cols) + 2,
        "features_note": "时域(7)×31列 + 频域FFT(3带)×4列 + 切削参数6 + 方向one-hot2",
        "split": split_desc,
        "imbalance_handling": "SMOTE(仅训练集)" if use_smote else "报告不平衡，训练时类别加权/SMOTE",
        "steps": [
            "label 解析（start/end/label + 切削参数）",
            "mtc.csv 时间切片对齐",
            "时域特征(7) + 频域特征(FFT 3 频带)",
            "近常数（低方差）列剔除",
            "Z-score 标准化（训练集拟合）",
            "train/val/test 划分（随机分层或按机床）",
        ],
    }
    _merge_index("msm", entry)


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
    print(f"[msm] 索引更新: {INDEX}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MSM 刀具状态识别预处理")
    ap.add_argument("--zip", default=DEFAULT_ZIP, help="MSM 数据集 zip 路径")
    ap.add_argument("--smote", action="store_true", help="对训练集做 SMOTE 过采样（需 imblearn）")
    ap.add_argument("--split-by-machine", action="store_true",
                    help="按机床划分（跨机泛化 held-out；默认随机分层，保证三类进 test）")
    ap.add_argument("--seed", type=int, default=42, help="随机分层划分种子")
    args = ap.parse_args()
    split_mode = "by_machine" if args.split_by_machine else "random"
    main(args.zip, args.smote, split_mode, args.seed)
