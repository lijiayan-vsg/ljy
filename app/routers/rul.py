# -*- coding: utf-8 -*-
"""剩余寿命（RUL）预测路由。"""
from fastapi import APIRouter

from .. import db, inference, samples, schemas

router = APIRouter()

RUL_WARN_THRESHOLD = 50.0  # 剩余寿命低于该值（min）时预警


@router.post("/predict/rul", response_model=schemas.RULResponse, summary="剩余寿命（RUL）预测")
def predict_rul(req: schemas.RULRequest):
    rul = inference.predict_rul(req.type, req.air_temperature, req.process_temperature,
                                req.rotational_speed, req.torque)
    warning = "刀具接近寿命极限，建议尽快换刀" if rul < RUL_WARN_THRESHOLD else None

    db.add_prediction(
        kind="rul",
        input_summary=(f"Type={req.type} 空气温度={req.air_temperature:.1f} "
                       f"转速={req.rotational_speed:.0f}"),
        result=f"RUL={rul:.1f} min",
        confidence=None,
    )
    return schemas.RULResponse(rul=round(rul, 2), warning=warning)


@router.get("/samples/rul", summary="获取 RUL 原始特征样例（含真实 RUL）")
def get_rul_samples():
    return samples.rul_samples()
