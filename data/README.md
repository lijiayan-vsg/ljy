# 数据来源与预处理说明

本目录存放本项目「CNC 铣削刀具磨损全生命周期预测与智能诊断系统」所用的原始数据、预处理程序及预处理后的数据文件。数据来源遵循课程设计阶段要求：**原始数据一律不提交仓库，仅提交有效来源链接与下载脚本；预处理后的特征矩阵提交到 `/data/processed`；不涉及私有敏感数据**。

## 目录结构

```
data/
├── README.md                # 本说明文件（数据来源 + 预处理说明）
├── download_data.py         # 数据集下载脚本（MSM / AI4I）
├── preprocess_msm.py        # MSM 数据预处理程序（状态识别，已完成）
├── preprocess.py            # AI4I 2020 数据预处理程序（RUL，已完成）
├── raw/                     # 原始数据（不入库，提交链接 + download_data.py 下载）
│   └── ai4i2020.csv         # AI4I 2020（10,000 行，约 522 KB，本地运行用）
└── processed/               # 预处理后特征矩阵 + 数据索引
    ├── msm_features.csv     # MSM 特征矩阵（2,570 段 × 237 特征 + 标签/划分）
    ├── msm_scaler.json      # MSM 标准化参数（推理复用）
    ├── ai4i_processed.csv   # AI4I 预处理后数据（10,000 行 + RUL 标签）
    ├── ai4i_scaler.json     # AI4I 标准化参数（推理复用）
    └── index.json           # 数据索引（来源、预处理步骤、标签分布等元数据）
```

## 一、数据来源

| 数据集 | 角色 | 用途 | 许可 | 获取方式 | 状态 |
|--------|------|------|------|----------|------|
| MSM（Multi-Sensor and MTConnect） | 主数据集 | 刀具状态识别（正常/异常/刀具缺陷三分类）+ 智能诊断 | CC BY 4.0 | Kaggle | ✅ 已下载 |
| AI4I 2020 预测性维护 | 辅助数据集 | 剩余寿命（RUL）预测 | CC BY 4.0 | UCI（提交下载链接） | ✅ 链接已登记 |

### 1. MSM（Multi-Sensor and MTConnect）数据集（主数据集）

- **全称**：Multi-sensor and MTConnect dataset of metal cutting anomaly in milling from laboratory and industry settings
- **用途**：本项目主训练/验证数据，用于刀具状态识别（正常切削 / 加工异常 / 刀具缺陷三分类）与智能诊断知识库构建。
- **下载链接**：https://www.kaggle.com/datasets/manufuturetoday/multi-sensor-for-metal-milling-anomaly
- **论文**：Kim, E., Sim, Y., Li, A.S. et al., *Scientific Data*, 2026. DOI: [10.1038/s41597-026-07255-7](https://doi.org/10.1038/s41597-026-07255-7)
- **许可**：CC BY 4.0
- **规模**：压缩 14.2 GB / 解压约 40 GB；3 台机床、69 次加工实验、2,575 个带标注的切片段。

#### 数据构成

三台机床：

- `imi_vm20i`（37 次实验）、`imi_vmx30ui`（19 次实验）：实验室 Hurco 铣床；
- `tmf_vf10`（13 次实验）：工业现场 Haas VF-10。

每个实验目录 `datasetN/` 下包含多模态文件：

| 文件 | 内容 | 采样/规模 | 备注 |
|------|------|-----------|------|
| `*_acc.csv` | 外部加速度计 2 通道（acc0, acc1） | ~50 kHz | 仅实验室机床，单文件约 309 MB |
| `*_ct.csv` | 电流互感器 1 通道 | 稀疏采样 | 仅工业机床 tmf_vf10 |
| `*_mtc.csv` | MTConnect 机床控制器 51 列 | ~6.4 Hz | 含主轴转速、各轴负载/位置、能量、振动 RMS 特征 |
| `*_s0/s1/s2.wav` | 声音信号 | 音频 | 2–3 路麦克风 |
| `*.mp4` | 加工视频 | — | 仅实验室机床 |
| `*.nc` | G 代码程序（非 NetCDF） | — | 首行注释注明实验意图（如 tool defect） |
| `label.csv` | 分段标注 | — | start/end/label + 切削参数 + 刀具 + 工件材料 |

顶层另含 `dataset_summary.xlsx`（69 行：machine/dataset/workpiece/tool/remark，remark 列含异常原因）与各机床的 `*.xml`（MTConnect 配置）。

#### 标签说明

`label.csv` 中 `label` 为三级标注，共 2,575 段：

| label | 含义 | 数量 | 占比 |
|-------|------|------|------|
| 0 | 正常切削 | 1,933 | 75.1% |
| 1 | 加工异常（需立即停机） | 465 | 18.1% |
| 2 | 刀具缺陷 | 177 | 6.9% |

> 存在明显类别不平衡（约 11:2.6:1），预处理阶段需采用 SMOTE / 类别加权等方式处理。

### 2. AI4I 2020 预测性维护数据集（辅助数据集，用于 RUL）

- **用途**：寿命预测（RUL）回归目标。含连续的 `Tool wear [min]` 磨损量字段及加工参数，可构造 RUL 目标训练 LSTM/随机森林模型。
- **来源**：UCI Machine Learning Repository（ID 601）
- **下载链接**：https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv
- **论文**：S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications", 2020.
- **许可**：CC BY 4.0
- **规模**：10,000 行 × 14 列，约 522 KB（提交下载链接，本地运行前 `python data/download_data.py ai4i` 获取）
- **关键字段**：`Air temperature [K]`、`Process temperature [K]`、`Rotational speed [rpm]`、`Torque [Nm]`、`Tool wear [min]`、`Machine failure`、`Type`（L/M/H）等。

## 二、数据预处理

### MSM（状态识别 + 诊断）—— `preprocess_msm.py`（已完成）

1. **label 解析**：解析 `label.csv`，得到切片段 `(start, end, label)` 及切削参数元数据；
2. **信号对齐**：将 `mtc.csv` 按时间戳切片到各标注段；
3. **特征提取**：对每段提取时域特征（均值、标准差、RMS、最小/最大、峭度、斜率）与频域特征（FFT 3 频带能量）；高频 `acc.csv`/`ct.csv` 及音频 `wav` 可按需追加；
4. **类别不平衡处理**：默认报告不平衡（训练时类别加权 / SMOTE），可选 `--smote` 仅对训练集过采样；
5. **归一化**：Z-score 标准化（StandardScaler，训练集拟合）；
6. **特征矩阵构建**：输出 `msm_features.csv`（2,570 段 × 237 特征）与 `msm_scaler.json`、`index.json` 索引（label 标注 2,575 段中 5 段落在 mtc.csv 采样间隙内被排除）；
7. **数据划分**：默认随机分层 70/15/15（三类均进 test），`--split-by-machine` 为跨机泛化 held-out。

运行方式：

```bash
python data/preprocess_msm.py --zip <MSM数据集zip路径> [--smote] [--split-by-machine]
```

> 预处理程序支持直接从 zip 内读取 `mtc.csv` + `label.csv`（合计约 100 MB）跑通全流程，无需解压 40 GB 高频信号。

### AI4I 2020（RUL）—— `preprocess.py`（已完成）

1. **缺失值填充**：数值列中位数 / 类别列众数；
2. **异常值处理**：IQR（1.5 倍四分位距）截断；
3. **特征工程**：构造温差、功率代理量等派生特征；
4. **标签构造**：由 `Tool wear [min]` 构造 RUL 回归目标；
5. **编码与归一化**：`Type` one-hot 编码、连续列 Z-score 标准化；
6. **输出**：预处理后 CSV 与 `index.json` 索引。

运行方式：

```bash
python data/preprocess.py
```

## 三、许可与引用

- MSM 数据集遵循 CC BY 4.0，引用：Kim, E., Sim, Y., Li, A.S. et al. *Multi-sensor and MTConnect dataset of metal cutting anomaly in milling from laboratory and industry settings*. Scientific Data 13, 07255 (2026).
- AI4I 2020 遵循 CC BY 4.0，引用：Matzka, S. *Explainable Artificial Intelligence for Predictive Maintenance Applications*. ETFA, 2020.
