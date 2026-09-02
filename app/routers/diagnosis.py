# -*- coding: utf-8 -*-
"""智能诊断（RAG）路由。"""
from fastapi import APIRouter, HTTPException

from .. import db, registry, schemas

router = APIRouter()


@router.post("/diagnosis/query", response_model=schemas.DiagnosisResponse, summary="RAG 诊断查询")
def diagnosis_query(req: schemas.DiagnosisRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")

    kb = registry.knowledge_base()
    hits = kb.query(req.query.strip(), top_k=req.top_k)
    results = [
        schemas.DiagnosisItem(
            status=d["status"], keywords=d["keywords"], cause=d["cause"],
            solution=d["solution"], similarity=round(s, 4),
        )
        for d, s in hits
    ]

    db.add_prediction(
        kind="diagnosis",
        input_summary=req.query.strip(),
        result="；".join(r.status for r in results) or "未匹配",
        confidence=results[0].similarity if results else None,
    )
    return schemas.DiagnosisResponse(results=results)


@router.get("/diagnosis/kb", summary="查看知识库全部条目")
def list_kb():
    kb = registry.knowledge_base()
    return kb.docs
