# 后端与前端（D8–D9）

FastAPI 后端 + Streamlit 前端，把 D7–D8 训练好的模型封装为可调用的 RESTful API 与交互看板。

## 目录结构

```
app/                          # FastAPI 后端
├── main.py                   # 应用入口（lifespan 初始化 DB + 预热模型）
├── config.py                 # 路径常量
├── schemas.py                # Pydantic 请求/响应模型
├── registry.py               # 模型 / scaler / 知识库单例加载
├── inference.py              # 推理逻辑（原始特征 → 标准化 → 预测）
├── samples.py                # 从 test 集反向还原原始特征样例
├── db.py                     # SQLite 预测历史记录
└── routers/                  # API 路由
    ├── status.py             # POST /api/predict/status · GET /api/samples/status
    ├── rul.py                # POST /api/predict/rul · GET /api/samples/rul
    ├── diagnosis.py          # POST /api/diagnosis/query · GET /api/diagnosis/kb
    └── history.py            # GET /api/history
web/
└── app.py                    # Streamlit 前端（总览/状态识别/寿命预测/诊断/历史）
tests/
└── test_api.py               # API 集成测试（TestClient 直连，可独立运行）
```

## 运行

在项目根目录 `d:/ljy` 下，先启动后端，再启动前端：

```bash
# 后端（端口 8000）
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 前端（另开终端）
python -m streamlit run web/app.py
```

- 后端 API 文档：http://127.0.0.1:8000/docs
- 前端界面：http://localhost:8501

### 集成测试

```bash
python tests/test_api.py
```

覆盖：健康检查、状态识别端到端（样例预测与真实标签一致性）、RUL 预测、RAG 诊断、非法输入拒绝（缺切削参数/信号 → 400、负 limit → 422）、历史记录。

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/predict/status` | 刀具状态三分类（方向 + 切削参数 + 229 信号特征） |
| GET | `/api/samples/status` | 状态识别原始特征样例（含真实标签，用于验证） |
| POST | `/api/predict/rul` | RUL 预测（Type + 4 个过程量） |
| GET | `/api/samples/rul` | RUL 原始特征样例（含真实 RUL） |
| POST | `/api/diagnosis/query` | RAG 诊断查询（文本 → Top-K 原因/方案） |
| GET | `/api/diagnosis/kb` | 知识库全部条目 |
| GET | `/api/history` | 预测历史记录 |

## 设计要点

- **端到端一致性**：后端接收「原始特征」，按训练时的同一套规则（方向 one-hot、切削参数透传、信号 Z-score、RUL 派生特征）还原为标准特征向量，特征顺序取自模型自带的 `feature_names_in_`，避免手工对齐出错。
- **防泄漏**：RUL 推理只用 Type + 空气/过程温度 + 转速 + 扭矩（派生温差/功率），不涉及磨损量本身。
- **样例验证**：`/api/samples/*` 把 test 集特征反向还原为原始特征，前端「预测样例」可与真实标签对比，验证「原始 → 标准化 → 预测」链路正确（状态识别 6/6 一致）。
- **输入校验**：缺失切削参数/信号特征返回 400（不静默填 0，避免产生分布外输入的错误诊断）；float 字段拒绝 NaN/Infinity（`allow_inf_nan=False` + 校验器）；历史 `limit` 用 `Query(ge=1, le=500)` 约束，防止负值/超大值异常返回。
- **历史入库**：每次预测/诊断自动写入 SQLite（`data/predictions.db`），前端「历史记录」页可追溯；连接启用 WAL + 超时 + 有限重试，规避 Windows 下偶发的文件锁只读错误。
