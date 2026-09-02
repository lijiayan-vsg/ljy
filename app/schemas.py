# -*- coding: utf-8 -*-
"""API 请求/响应 Pydantic 模型。"""
import math
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------- 刀具状态识别 ----------
class StatusRequest(BaseModel):
    """原始特征输入：走刀方向 + 切削参数 + 229 个原始信号特征。"""
    direction: str = Field(..., description="走刀方向：UP / DOWN")
    params: Dict[str, float] = Field(..., description="切削参数 N/ap/ae/F/Z/D")
    signals: Dict[str, float] = Field(..., description="229 个原始信号特征值（键为信号列名）")

    @field_validator("params", "signals")
    @classmethod
    def _reject_nonfinite(cls, v: Dict[str, float]) -> Dict[str, float]:
        for k, x in v.items():
            if math.isnan(x) or math.isinf(x):
                raise ValueError(f"{k} 不允许 NaN/Infinity")
        return v


class StatusResponse(BaseModel):
    label: int = Field(..., description="预测类别 0/1/2")
    label_name: str = Field(..., description="类别中文名")
    confidence: float = Field(..., description="预测置信度（最高类概率）")
    probabilities: Dict[str, float] = Field(..., description="各类别概率")


# ---------- 剩余寿命预测 ----------
class RULRequest(BaseModel):
    type: str = Field("M", description="产品类型：L / M / H")
    air_temperature: float = Field(..., allow_inf_nan=False, description="空气温度 [K]")
    process_temperature: float = Field(..., allow_inf_nan=False, description="过程温度 [K]")
    rotational_speed: float = Field(..., allow_inf_nan=False, description="转速 [rpm]")
    torque: float = Field(..., allow_inf_nan=False, description="扭矩 [Nm]")


class RULResponse(BaseModel):
    rul: float = Field(..., description="剩余寿命（RUL，单位：min）")
    warning: Optional[str] = Field(None, description="低于阈值时的预警提示")


# ---------- 智能诊断（RAG） ----------
class DiagnosisRequest(BaseModel):
    query: str = Field(..., description="诊断查询文本")
    top_k: int = Field(3, ge=1, le=10, description="返回 Top-K 条")


class DiagnosisItem(BaseModel):
    status: str
    keywords: str
    cause: str
    solution: str
    similarity: float


class DiagnosisResponse(BaseModel):
    results: List[DiagnosisItem]


# ---------- 历史记录 ----------
class HistoryItem(BaseModel):
    id: int
    ts: str
    kind: str
    input_summary: str
    result: str
    confidence: Optional[float] = None
