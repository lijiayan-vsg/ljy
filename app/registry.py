# -*- coding: utf-8 -*-
"""模型 / scaler / 知识库的单例加载器（首次调用加载，之后复用）。"""
import json
import threading

import joblib

from . import config

_lock = threading.Lock()
_state: dict = {"loaded": False}


def _load_all():
    _state["clf"] = joblib.load(config.STATUS_MODEL)          # 状态三分类
    _state["reg"] = joblib.load(config.RUL_MODEL)             # RUL 回归
    _state["msm_scaler"] = json.load(open(config.MSM_SCALER, encoding="utf-8"))
    _state["ai4i_scaler"] = json.load(open(config.AI4I_SCALER, encoding="utf-8"))
    _state["loaded"] = True


def ensure_loaded() -> dict:
    """确保模型已加载，返回内部状态 dict（含 clf/reg/msm_scaler/ai4i_scaler）。"""
    if not _state["loaded"]:
        with _lock:
            if not _state["loaded"]:
                _load_all()
    return _state


def knowledge_base() -> "KnowledgeBase":
    """延迟构建 RAG 知识库（TF-IDF 向量化一次后复用）。"""
    if "kb" not in _state:
        with _lock:
            if "kb" not in _state:
                from src.rag import KnowledgeBase, load_knowledge_base
                _state["kb"] = KnowledgeBase(load_knowledge_base())
    return _state["kb"]
