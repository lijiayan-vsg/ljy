# -*- coding: utf-8 -*-
"""刀具状态识别路由。"""
from fastapi import APIRouter, HTTPException

from .. import db, inference, samples, schemas

router = APIRouter()


@router.post("/predict/status", response_model=schemas.StatusResponse, summary="刀具状态三分类预测")
def predict_status(req: schemas.StatusRequest):
    try:
        label, name, proba, conf = inference.predict_status(req.direction, req.params, req.signals)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    db.add_prediction(
        kind="status",
        input_summary=f"方向={req.direction} 参数={req.params}",
        result=f"{name}({label})",
        confidence=conf,
    )
    return schemas.StatusResponse(
        label=label, label_name=name, confidence=conf, probabilities=proba,
    )


@router.get("/samples/status", summary="获取状态识别原始特征样例（含真实标签）")
def get_status_samples():
    return samples.status_samples()
