# -*- coding: utf-8 -*-
"""后端全局路径常量。"""
import os

# 项目根目录 = app/ 的上一级
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")
PROCESSED = os.path.join(DATA_DIR, "processed")
SRC_DIR = os.path.join(ROOT, "src")

# 模型与 scaler
STATUS_MODEL = os.path.join(MODELS_DIR, "status_classifier.joblib")
RUL_MODEL = os.path.join(MODELS_DIR, "rul_model.joblib")
MSM_SCALER = os.path.join(PROCESSED, "msm_scaler.json")
AI4I_SCALER = os.path.join(PROCESSED, "ai4i_scaler.json")

# 数据与知识库
MSM_FEATURES = os.path.join(PROCESSED, "msm_features.csv")
AI4I_PROCESSED = os.path.join(PROCESSED, "ai4i_processed.csv")
KNOWLEDGE_BASE = os.path.join(SRC_DIR, "knowledge_base.json")

# SQLite 历史记录库
DB_PATH = os.path.join(DATA_DIR, "predictions.db")

# 刀具状态标签
STATUS_LABELS = {0: "正常切削", 1: "加工异常", 2: "刀具缺陷"}
