# 算法模块（D7–D8）

本项目核心算法模块，覆盖《方案设计.md》中规划的四个算法模块：**遗传算法超参数寻优、刀具状态识别、剩余寿命（RUL）预测、RAG 知识库检索**，全部仅依赖 `numpy / pandas / scikit-learn / joblib`（不引入 PyTorch、xgboost、chromadb 等重依赖，见「实现适配说明」）。

## 目录结构

```
src/
├── __init__.py             # 包声明
├── paths.py                # 全局路径常量（数据/模型/知识库）
├── ga.py                   # 遗传算法超参数寻优（手写，仅 numpy）
├── classifier.py           # 刀具状态三分类（随机森林/梯度提升 + GA）
├── rul.py                  # RUL 回归预测（随机森林/梯度提升/MLP + GA）
├── rag.py                  # RAG 知识库检索（TF-IDF + 余弦相似度）
└── knowledge_base.json     # 诊断知识库（状态→原因→处置方案）
```

## 各模块说明

### 1. `ga.py` —— 遗传算法超参数寻优（智能优化技术）

从零实现标准遗传算法，不依赖 `deap`：

- **编码**：实数 / 整数 / 离散（choice，可为 `None`、字符串）三类参数混合编码；
- **算子**：锦标赛选择 + 单点交叉 + 逐基因变异 + 精英保留（elitism）；
- **适应度缓存**：相同个体只评估一次，避免重复训练。

```python
from src.ga import GeneticOptimizer
space = {"n_estimators": {"type": "int", "low": 80, "high": 260},
         "max_depth": {"type": "choice", "values": [None, 6, 10, 14, 20]}}
ga = GeneticOptimizer(space, fitness_fn, pop_size=10, n_generations=6, seed=42)
best_params, best_fitness = ga.run()
```

### 2. `classifier.py` —— 刀具状态三分类（模式与图像识别）

对 MSM 特征矩阵做「正常(0) / 加工异常(1) / 刀具缺陷(2)」三分类。

- **模型**：随机森林（主）/ 梯度提升（备选），`class_weight='balanced'` 处理类别不平衡；
- **超参数**：GA 寻优，适应度 = **3 折分层交叉验证 macro-F1**（比单一验证集更稳健）；
- **划分**：调参/训练用 train+val 合并，test 全程独立仅做最终评估。

```bash
python -m src.classifier            # 随机森林 + GA 寻优
python -m src.classifier --model gb # 梯度提升 + GA
python -m src.classifier --no-ga    # 快速基线
```

**结果（RF + GA，test 集 386 段）**：accuracy = **96.11%**，macro-F1 = **0.9444**，weighted-F1 = 0.9617。混淆矩阵（行=真实，列=预测）：

| 真实\预测 | 正常(0) | 异常(1) | 缺陷(2) |
|-----------|---------|---------|---------|
| 正常(0) 289 | 278 | 9 | 2 |
| 异常(1) 70  | 3   | 66 | 1 |
| 缺陷(2) 27  | 0   | 0  | 27 |

### 3. `rul.py` —— 剩余寿命（RUL）预测（寿命预测）

对 AI4I 2020 预处理数据回归预测 RUL（`RUL = max(0, 253 - Tool wear)`）。

- **模型**：随机森林（主，GA 寻优）/ 梯度提升（GA 寻优）/ MLP（神经网络基线）；
- **超参数**：GA 寻优，适应度 = 3 折交叉验证 neg-MAE；
- **防泄漏关键设计**：RUL 由磨损量直接构造，故磨损量本身（`Tool wear [min]` 与 `wear_raw`）已从特征中剔除，仅用加工过程信号（温度/扭矩/转速/类型/派生特征）推断 RUL。

```bash
python -m src.rul                     # 随机森林 + GA 寻优
python -m src.rul --model gbr         # 梯度提升 + GA
python -m src.rul --model mlp --no-ga # MLP 神经网络
```

**结果（test 集 1,500 行，RUL 范围 0–253）**：

| 模型 | MAE (min) | RMSE (min) | R² |
|------|-----------|------------|-----|
| 随机森林（GA 寻优） | **49.53** | **60.79** | **0.0933** |
| 梯度提升（GA 寻优） | 50.61 | 61.46 | 0.0734 |
| 随机森林（基线） | 49.54 | 61.22 | 0.0806 |
| MLP 神经网络 | 55.34 | 63.99 | -0.0045 |

> **关键发现（如实记录）**：AI4I 2020 的 `Tool wear [min]` 是与加工参数**解耦的独立随机变量**（数据集生成设计即如此），故仅凭加工过程信号无法强预测磨损/RUL（R²≈0.09）。这并非模型缺陷，而是数据特性 —— 它说明真实刀具寿命预测必须依赖与磨损强相关的传感器（振动、主轴负载、声发射等），这正是本项目状态识别模块采用 MSM 多传感器数据的原因。该模块完整演示了「回归 + GA 寻优 + 交叉验证 + 特征重要性」全流程。

### 4. `rag.py` —— 知识库检索增强（知识工程）

对诊断知识库做向量检索，异常检测后自动召回诊断建议。

- **向量化**：TF-IDF + 字符 n-gram（`char_wb`，2~3 元），无需分词器即可处理中文查询；
- **检索**：余弦相似度召回 Top-K，返回「状态 → 可能原因 → 处置方案」；
- **抽象接口**：`KnowledgeBase` 类已独立封装，后续可无缝替换为 ChromaDB 持久化向量库。

```bash
python -m src.rag                      # 演示多条查询
python -m src.rag --query "主轴振动异常" --top-k 3
```

知识库内容见 `knowledge_base.json`（10 条：正常切削 + 主轴过载/颤振/噪声/温升/积屑瘤/崩刃/磨损/断裂/表面粗糙等异常与缺陷条目）。

## 模型产物（`models/`）

训练脚本自动生成：`status_classifier.joblib`、`rul_model.joblib` 及对应的 `*_best_params.json`（GA 最优参数）与 `*_metrics.json`（评估指标）。模型可由脚本随时重新训练生成，故 `.joblib` 大文件不提交 git（见 `.gitignore`）。

## 实现适配说明（与《方案设计.md》的差异）

| 方案设计原规划 | 实际实现 | 原因 |
|----------------|----------|------|
| 状态分类：随机森林 / XGBoost | 随机森林 / 梯度提升（sklearn） | xgboost 未安装；sklearn GradientBoosting 等价，避免重依赖 |
| 寿命预测：LSTM（PyTorch） | 随机森林 / 梯度提升 / MLP（sklearn） | AI4I 为截面表格数据（非时序），LSTM 不适用；torch 未安装；MLP 提供神经网络基线 |
| 超参数优化：遗传算法 | 手写遗传算法（numpy） | deap 未安装；从零实现更完整展示算法原理 |
| 知识库：ChromaDB | TF-IDF + 余弦相似度（sklearn） | chromadb 重依赖；轻量向量检索已抽象接口，可替换 ChromaDB |

> LSTM 时序建模适合 MSM 的 50 kHz 振动/6.4 Hz 控制器时序数据，若后续接入 PyTorch 可作为扩展方向。
