# -*- coding: utf-8 -*-
"""
FastAPI 应用入口。

运行（在项目根目录 d:/ljy 下）：
    uvicorn app.main:app --host 0.0.0.0 --port 8000

启动后访问 http://127.0.0.1:8000/docs 查看自动生成的 API 文档。
"""
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import db, registry
from .routers import diagnosis, history, rul, status


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：初始化数据库并预热模型（避免首个请求冷启动）
    db.init_db()
    registry.ensure_loaded()
    yield


app = FastAPI(
    title="CNC 铣削刀具磨损全生命周期预测与智能诊断 API",
    description="刀具状态三分类 / 剩余寿命预测 / RAG 智能诊断 / 预测历史。",
    version="0.1.0",
    lifespan=lifespan,
)

# 开发期放开跨域（前端 Streamlit 与后端分离部署）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status.router, prefix="/api", tags=["状态识别"])
app.include_router(rul.router, prefix="/api", tags=["寿命预测"])
app.include_router(diagnosis.router, prefix="/api", tags=["智能诊断"])
app.include_router(history.router, prefix="/api", tags=["历史记录"])


def _sanitize_nonfinite(obj):
    """递归把错误详情里的 NaN/Infinity 转成字符串，避免 JSON 序列化崩溃。"""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return repr(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nonfinite(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # 请求体含 NaN/Infinity 时，默认 422 详情里的 input 值无法 JSON 序列化，这里先清洗
    return JSONResponse(status_code=422, content={"detail": _sanitize_nonfinite(exc.errors())})


@app.get("/api/health", summary="健康检查")
def health():
    return {"status": "ok", "model": "status_classifier + rul_model 已加载"}
