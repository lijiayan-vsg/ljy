# CNC 铣削刀具磨损全生命周期预测与智能诊断系统

基于多传感器信号（振动、声音、切削电流、机床控制器数据等）的 CNC 铣削刀具状态识别、剩余使用寿命（RUL）预测与智能诊断系统。本项目为《制造智能技术》课程设计，采用 vibe coding 方法（Claude Code 为核心 Harness 工具）完成。

## 项目目标

- **刀具状态识别**：对多传感器信号进行时频域特征提取与分类，识别正常切削、加工异常、刀具缺陷三类状态。
- **刀具剩余寿命预测**：基于历史磨损数据建立 RUL 预测模型，在接近报废阈值前预警。
- **智能诊断与决策支持**：构建状态—异常原因—解决方案知识库，检测到异常时通过 RAG 自动推送诊断建议。

技术方向覆盖：**智能优化技术**（遗传算法超参数寻优）、**模式与图像识别**（时频域特征 + 分类器）、**知识工程**（知识库 + RAG）。

## 项目结构

```
├── readme.md                 # 本说明文件
├── 选题说明.md                # 选题与目标说明
├── 方案设计.md                # 详细方案设计（技术栈/架构/技术路线/计划）
├── 需求规格说明书.md           # 需求规格说明书（功能/接口/数据/验收标准）
├── 设计报告.md                # 设计报告（背景/方案/实现/测试/AI 使用披露/总结）
├── 答辩PPT大纲.md             # 答辩 PPT 大纲与问答准备
├── 演示视频脚本.md            # 3 分钟演示视频脚本
├── 提交清单.md                # 课设提交物清单与自检
├── 学习笔记.md                # vibe coding / Git / 选题调研学习记录
├── data/                     # 数据来源与预处理
│   ├── README.md             # 数据来源说明（含有效下载链接与许可）
│   ├── download_data.py      # 数据集下载脚本
│   ├── preprocess_msm.py     # MSM 数据预处理程序（状态识别，已完成）
│   ├── preprocess.py         # AI4I 2020 数据预处理程序（RUL，已完成）
│   ├── raw/                  # 原始数据（不入库，提交链接 + 下载脚本）
│   │   └── ai4i2020.csv      # AI4I 2020（约 522 KB，本地运行用）
│   └── processed/            # 预处理后特征矩阵 + 数据索引
│       ├── msm_features.csv / msm_scaler.json
│       ├── ai4i_processed.csv / ai4i_scaler.json
│       └── index.json
├── src/                      # 算法模块（D7-D8，已完成）
│   ├── README.md             # 算法模块说明（含评估结果与实现适配说明）
│   ├── ga.py                 # 遗传算法超参数寻优（手写，仅 numpy）
│   ├── classifier.py         # 刀具状态三分类（随机森林/梯度提升 + GA）
│   ├── rul.py                # RUL 预测（随机森林/梯度提升/MLP + GA）
│   ├── rag.py                # RAG 知识库检索（TF-IDF + 余弦相似度）
│   └── knowledge_base.json   # 诊断知识库（状态→原因→处置方案）
├── app/                      # FastAPI 后端（D8-D9，已完成）
│   ├── README.md             # 后端/前端运行说明与 API 一览
│   ├── main.py               # 应用入口（lifespan 初始化 DB + 预热模型）
│   ├── inference.py          # 推理逻辑（原始特征 → 标准化 → 预测）
│   ├── samples.py            # 样例生成（test 集反向还原原始特征）
│   ├── db.py                 # SQLite 预测历史记录
│   └── routers/              # status / rul / diagnosis / history 路由
├── web/
│   └── app.py                # Streamlit 前端看板（总览/状态/寿命/诊断/历史）
├── tests/
│   └── test_api.py           # API 集成测试（TestClient 直连，可独立运行）
├── models/                   # 训练模型产物（*.joblib 可由脚本重建，不提交 git）
└── prompt/                   # AI 工具提示词追溯记录（每阶段更新）
    └── *.json
```

## 数据来源

数据来源遵循课程设计要求：**原始数据一律不提交仓库，仅提交有效来源链接与下载脚本；预处理后的特征矩阵提交到 `/data/processed`；不涉及私有敏感数据**。详见 [data/README.md](data/README.md)。

| 数据集 | 用途 | 许可 | 链接 | 状态 |
|--------|------|------|------|------|
| MSM（Multi-Sensor and MTConnect） | 状态识别 + 智能诊断（三级标注：正常/异常/刀具缺陷） | CC BY 4.0 | [Kaggle](https://www.kaggle.com/datasets/manufuturetoday/multi-sensor-for-metal-milling-anomaly) · [论文](https://doi.org/10.1038/s41597-026-07255-7) | ✅ 已下载（14.2 GB） |
| AI4I 2020 预测性维护 | 剩余寿命（RUL）预测（含连续 Tool wear 字段） | CC BY 4.0 | [UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv) | ✅ 链接已登记 |

> 注：两个数据集的原始数据均不提交到 git 仓库，仅在 `/data` 记录有效来源链接并提供 `data/download_data.py` 下载脚本；预处理后的特征矩阵提交到 `data/processed/`。

## 数据预处理

预处理程序实现完整流程：**label 分段解析 → 多传感器信号对齐 → 时域/频域特征提取 → 类别不平衡处理 → 归一化 → 特征矩阵构建**，并输出特征矩阵与数据索引 `index.json`。

- `data/preprocess_msm.py`：MSM 状态识别与诊断的预处理（已完成，输出 msm_features.csv，2,570 段 × 237 特征）；
- `data/preprocess.py`：AI4I 2020 寿命预测（RUL）的预处理（已完成，输出 ai4i_processed.csv）。

```bash
python data/preprocess_msm.py --zip <MSM数据集zip路径>
python data/preprocess.py
```

## AI 工具提示词追溯

本项目使用 Claude Code 作为核心 Harness 工具，AI 交流记录以 JSON 形式存放于 [prompt/](prompt/) 目录，按阶段同步更新，并在上下文压缩前及时备份。

## 算法模块

四个核心算法模块已实现（详见 [src/README.md](src/README.md)），仅依赖 numpy/pandas/scikit-learn/joblib：

| 模块 | 文件 | 结果 |
|------|------|------|
| 遗传算法超参数寻优 | [src/ga.py](src/ga.py) | 手写 GA（锦标赛+单点交叉+变异+精英保留），适应度=3 折 CV |
| 刀具状态三分类 | [src/classifier.py](src/classifier.py) | 随机森林 + GA，test 准确率 **96.11%**、macro-F1 **0.9444** |
| RUL 预测 | [src/rul.py](src/rul.py) | 随机森林 + GA，test MAE **49.53** min、R² **0.0933**（见下） |
| RAG 知识库 | [src/rag.py](src/rag.py) | TF-IDF + 余弦相似度，Top-K 诊断召回 |

> **RUL 关键发现**：AI4I 2020 的磨损量是与加工参数解耦的独立随机变量，仅凭过程信号无法强预测 RUL（R²≈0.09）——这是数据特性而非模型缺陷，说明真实寿命预测须依赖磨损强相关的传感器（MSM 振动/主轴负载）。详见 [src/README.md](src/README.md)。

## 后端与前端

FastAPI 后端封装训练好的模型为 RESTful API，Streamlit 前端提供交互看板（总览 / 状态识别 / 寿命预测 / 智能诊断 / 历史记录）。详见 [app/README.md](app/README.md)。

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000   # 后端
python -m streamlit run web/app.py                          # 前端
python tests/test_api.py                                    # 集成测试（TestClient 直连）
```

API 文档：http://127.0.0.1:8000/docs

## 技术栈

前端 Streamlit · 后端 FastAPI · 数据库 SQLite · 算法 Scikit-learn（随机森林/梯度提升/MLP）+ 手写遗传算法 + TF-IDF 向量检索（PyTorch/ChromaDB 为可选扩展） · 版本控制 Git。详见 [方案设计.md](方案设计.md)。

## 进度

| 阶段 | 状态 |
|------|------|
| D1–D2 工具配置 / vibe coding 学习 / 选题调研 | ✅ 完成 |
| D3–D5 选题与方案设计 | ✅ 完成 |
| D6 数据来源、清洗、预处理与入库 | ✅ 完成 |
| D7–D8 算法模块开发 | ✅ 完成 |
| D8–D9 后端与前端开发 | ✅ 完成 |
| D9–D10 集成调试 / 报告 / 答辩 | ✅ 报告与文档完成（视频 / PPT 待按大纲录制制作） |
