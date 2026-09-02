# -*- coding: utf-8 -*-
"""全局路径常量：数据、模型、知识库目录（供各算法模块复用）。"""
import os

# 项目根目录 = src/ 的上一级
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
PROCESSED = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT, "models")
SRC_DIR = os.path.join(ROOT, "src")

# 预处理后特征矩阵
MSM_FEATURES = os.path.join(PROCESSED, "msm_features.csv")     # 状态识别（三分类）
AI4I_PROCESSED = os.path.join(PROCESSED, "ai4i_processed.csv")  # RUL 回归

# 知识库
KNOWLEDGE_BASE = os.path.join(SRC_DIR, "knowledge_base.json")

os.makedirs(MODELS_DIR, exist_ok=True)
