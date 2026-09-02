# -*- coding: utf-8 -*-
"""预测历史记录路由。"""
from typing import List

from fastapi import APIRouter, Query

from .. import db, schemas

router = APIRouter()


@router.get("/history", response_model=List[schemas.HistoryItem], summary="查询预测历史")
def get_history(limit: int = Query(50, ge=1, le=500, description="返回条数（1–500）")):
    return db.list_predictions(limit=limit)
