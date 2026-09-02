# -*- coding: utf-8 -*-
"""
API 集成测试（无需外部服务器，用 FastAPI TestClient 直连 ASGI）。

运行（项目根目录 d:/ljy 下）：
    python tests/test_api.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Windows 控制台默认 GBK，强制 UTF-8 避免中文/符号编码报错
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient

from app.main import app


def test_all():
    with TestClient(app) as c:
        # 1. 健康检查
        assert c.get("/api/health").json()["status"] == "ok"

        # 2. 状态识别：样例 → 预测，必须与真实标签一致（端到端一致性）
        samples = c.get("/api/samples/status").json()
        assert len(samples) >= 3
        n_ok = 0
        for s in samples:
            r = c.post("/api/predict/status", json={
                "direction": s["direction"], "params": s["params"], "signals": s["signals"]})
            assert r.status_code == 200, r.text
            d = r.json()
            assert d["label"] == s["label"], f"{s['id']} 预测 {d['label_name']} != 真实 {s['label_name']}"
            assert 0 <= d["confidence"] <= 1
            n_ok += 1
        print(f"[test] 状态识别 {n_ok}/{len(samples)} 一致 ✓")

        # 3. RUL：样例 → 预测，应返回非负有限值
        rs = c.get("/api/samples/rul").json()
        assert len(rs) >= 3
        for s in rs:
            r = c.post("/api/predict/rul", json={
                "type": s["type"], "air_temperature": s["air_temperature"],
                "process_temperature": s["process_temperature"],
                "rotational_speed": s["rotational_speed"], "torque": s["torque"]})
            assert r.status_code == 200, r.text
            assert r.json()["rul"] >= 0
        print(f"[test] RUL 预测 {len(rs)} 条非负 ✓")

        # 3.5 NaN/Infinity 应被拒绝（422），不得静默返回伪造 RUL 或触发 500
        # httpx 的 json= 会拒绝 NaN，故发原始体（requests 客户端 json.dumps 默认可发出 NaN）
        for bad in ("NaN", "Infinity", "-Infinity"):
            body = ('{"type": "M", "air_temperature": %s, "process_temperature": 300.0,'
                    ' "rotational_speed": 1500.0, "torque": 40.0}') % bad
            r = c.post("/api/predict/rul", content=body, headers={"Content-Type": "application/json"})
            assert r.status_code == 422, f"NaN/Inf 应返回 422，实际 {r.status_code}"
        print("[test] RUL NaN/Infinity 拒绝 ✓")

        # 4. 诊断：中文查询返回结果
        d = c.post("/api/diagnosis/query", json={"query": "刀具崩刃", "top_k": 3}).json()
        assert len(d["results"]) > 0
        assert all("solution" in it for it in d["results"])
        print(f"[test] RAG 诊断 top1={d['results'][0]['status']} ✓")

        # 5. 缺特征/缺切削参数应返回 400（不得静默填 0）
        r = c.post("/api/predict/status", json={"direction": "UP", "params": {}, "signals": {}})
        assert r.status_code == 400, "缺少切削参数应返回 400"
        r = c.post("/api/predict/status",
                   json={"direction": "UP", "params": {"N": 1000, "ap": 0.5, "ae": 0.5,
                                                         "F": 0.1, "Z": 2, "D": 10}, "signals": {}})
        assert r.status_code == 400, "缺少信号特征应返回 400"

        # 6. 历史记录可查询；非法 limit 应被拒绝
        h = c.get("/api/history").json()
        assert isinstance(h, list)
        assert c.get("/api/history", params={"limit": -1}).status_code == 422, "负 limit 应返回 422"
        assert c.get("/api/history", params={"limit": 0}).status_code == 422, "0 limit 应返回 422"
        print(f"[test] 历史记录 {len(h)} 条 ✓")

    print("\n全部集成测试通过 ✓")


if __name__ == "__main__":
    test_all()
