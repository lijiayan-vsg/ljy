# -*- coding: utf-8 -*-
"""
数据集下载脚本（阶段1：数据来源）

本项目使用两个数据集：
  1. MSM（主数据集）：多传感器金属铣削异常数据集，来源 Kaggle（CC BY 4.0）；
  2. AI4I 2020（辅助数据集）：预测性维护数据集，来源 UCI（CC BY 4.0，小数据已直接提交）。

用法：
    python data/download_data.py msm       # 通过 Kaggle API 下载 MSM（需配置 kaggle.json）
    python data/download_data.py ai4i      # 从 UCI 直链下载 AI4I 2020（无需配置）
"""
import argparse
import os
import subprocess
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, "raw")

KAGGLE_SLUG = "manufuturetoday/multi-sensor-for-metal-milling-anomaly"
DATASET_URL = "https://www.kaggle.com/datasets/manufuturetoday/multi-sensor-for-metal-milling-anomaly"

AI4I_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"


def download_msm() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[msm] 从 Kaggle 下载 MSM 数据集: {DATASET_URL}")
    print("  前置条件：生成 Kaggle API Token 并配置 kaggle.json 到 ~/.kaggle/")
    try:
        subprocess.check_call(
            ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", RAW_DIR, "--unzip"]
        )
        print(f"[msm] 下载完成: {RAW_DIR}")
    except FileNotFoundError:
        print("  未检测到 kaggle CLI，请先执行: pip install kaggle 并配置 kaggle.json")
    except subprocess.CalledProcessError as e:
        print(f"  下载失败: {e}")
        print(f"  也可手动在浏览器下载: {DATASET_URL}")


def download_ai4i() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    out = os.path.join(RAW_DIR, "ai4i2020.csv")
    print(f"[ai4i] 从 UCI 下载 AI4I 2020: {AI4I_URL}")
    urllib.request.urlretrieve(AI4I_URL, out)
    print(f"[ai4i] 下载完成: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载数据集")
    parser.add_argument("dataset", choices=["msm", "ai4i"], help="要下载的数据集")
    args = parser.parse_args()
    if args.dataset == "msm":
        download_msm()
    else:
        download_ai4i()


if __name__ == "__main__":
    main()
