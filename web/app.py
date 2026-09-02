# -*- coding: utf-8 -*-
"""
Streamlit 前端：CNC 铣削刀具磨损全生命周期预测与智能诊断系统。

运行（先启动后端，再在项目根目录 d:/ljy 下）：
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    streamlit run web/app.py
"""
import json
import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_BASE", "http://127.0.0.1:8000")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="刀具磨损预测与诊断", page_icon="🔧", layout="wide")

# ============ 全局样式 ============
st.markdown(
    """
    <style>
    /* 基础字体与背景：中性浅灰（去除蓝色调） */
    html, body, .stApp { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
    .stApp { background: #f5f6f8; }

    /* 去掉顶部渐变条与工具栏，收紧上下留白 */
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stToolbar"] { display: none; }
    footer { display: none; }
    .block-container { padding-top: 1rem; padding-bottom: 1.6rem; max-width: 1320px; }

    /* 侧边栏：炭黑钢色（非蓝） */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16181d 0%, #1d2026 100%);
    }
    [data-testid="stSidebar"] h2 { color: #ffffff; }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span { color: #c7ccd4; }
    /* 功能导航选项文字：纯白加粗，确保深色背景上清晰可读 */
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] label div {
        color: #ffffff !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] hr { border-color: #2c3038; }

    /* 指标卡片：数值用等宽字体，呈现仪表读数感 */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e6ea;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 1px 2px rgba(16,24,40,.05);
    }
    [data-testid="stMetricLabel"] { color: #5c6570; }
    [data-testid="stMetricValue"] {
        color: #d97706;
        font-family: "Cascadia Mono", Consolas, monospace;
    }

    /* 主按钮：琥珀色 */
    .stButton > button {
        border-radius: 10px; border: none; background: #b45309;
        color: #fff; font-weight: 600; padding: 0.5rem 1.4rem;
    }
    .stButton > button:hover { background: #92400e; color: #fff; }

    /* 标题配色 */
    h1, h2, h3 { color: #1c2025; }

    /* Hero：炭黑面板 + 琥珀左竖线（无蓝紫渐变） */
    .hero {
        background: #181b20;
        border: 1px solid #2a2e36;
        border-left: 4px solid #d97706;
        border-radius: 10px;
        padding: 1.6rem 2rem;
        color: #eef1f5;
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: #ffffff; margin: 0 0 0.4rem 0; font-size: 1.75rem; }
    .hero p { color: #b7bdc7; margin: 0; font-size: 0.98rem; }
    .eyebrow {
        color: #d97706; font-size: 0.78rem; letter-spacing: 0.08em;
        font-weight: 600; margin-bottom: 0.4rem;
    }

    /* 分区标题（左侧琥珀竖线） */
    .sec {
        border-left: 4px solid #d97706; padding-left: 0.7rem;
        margin: 1rem 0 0.6rem 0;
    }
    .sec h2 { margin: 0; font-size: 1.25rem; }
    .sec p { margin: 0.15rem 0 0 0; color: #5c6570; }

    /* 状态徽章 + 状态圆点（替代 emoji 指示灯） */
    .badge {
        display: inline-block; background: #2a2e36;
        border: 1px solid #3a3f49; border-radius: 999px;
        padding: 0.15rem 0.8rem; font-size: 0.82rem; margin-left: 0.6rem;
        color: #eef1f5;
    }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
    .dot-ok { background: #22c55e; }
    .dot-off { background: #ef4444; }

    /* 系统流程图 */
    .flow { display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: stretch; }
    .flow-step {
        flex: 1 1 150px; background: #ffffff; border: 1px solid #e4e6ea;
        border-left: 3px solid #d97706; border-radius: 10px; padding: 0.8rem 0.9rem;
    }
    .flow-step .n { color: #d97706; font-family: "Cascadia Mono", Consolas, monospace; font-size: 0.72rem; }
    .flow-step .t { font-weight: 600; color: #1c2025; margin-top: 0.2rem; font-size: 0.95rem; }
    .flow-step .d { color: #5c6570; font-size: 0.8rem; margin-top: 0.2rem; line-height: 1.4; }
    .flow-arrow { align-self: center; color: #d97706; font-size: 1.15rem; font-weight: 600; }

    /* 数据源 / 说明信息块 */
    .panel {
        background: #ffffff; border: 1px solid #e4e6ea; border-radius: 10px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
    }
    .panel .k { color: #5c6570; font-size: 0.8rem; }
    .panel .v { font-family: "Cascadia Mono", Consolas, monospace; color: #1c2025; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============ API 辅助 ============
def api(path: str, method: str = "GET", **kwargs):
    """统一请求，后端未启动时给出友好提示。"""
    try:
        r = requests.request(method, f"{API}{path}", timeout=15, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error(f"无法连接后端 `{API}`，请先启动：`uvicorn app.main:app --port 8000`")
    except requests.HTTPError as e:
        st.error(f"接口错误：{e.response.status_code} {e.response.text[:200]}")
    return None


@st.cache_data(ttl=30)
def backend_ok() -> bool:
    try:
        return requests.get(f"{API}/api/health", timeout=3).status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=60)
def load_json(relpath: str):
    """读取项目内的模型指标 / 数据索引 JSON，失败返回 None。"""
    try:
        with open(os.path.join(ROOT, relpath), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def header(title: str, subtitle: str = ""):
    """统一的紧凑型页面标题（左侧琥珀竖线 + 副标题）。"""
    st.markdown(
        f'<div class="sec"><h2>{title}</h2><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


STATUS_LABELS = ["正常切削", "加工异常", "刀具缺陷"]


def per_class_table(cm):
    """由混淆矩阵推导每类 precision / recall / F1。"""
    rows = []
    for i in range(len(cm)):
        tp = cm[i][i]
        col = sum(cm[k][i] for k in range(len(cm)))
        row = sum(cm[i][k] for k in range(len(cm)))
        p = tp / col if col else 0.0
        r = tp / row if row else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append({"类别": STATUS_LABELS[i], "样本": row, "Precision": p, "Recall": r, "F1": f1})
    return pd.DataFrame(rows)


def flow_html(steps):
    """steps: [(序号, 标题, 说明)]，渲染横向流程条。"""
    parts = []
    for i, (n, t, d) in enumerate(steps):
        if i > 0:
            parts.append('<div class="flow-arrow">→</div>')
        parts.append(
            f'<div class="flow-step"><div class="n">{n}</div>'
            f'<div class="t">{t}</div><div class="d">{d}</div></div>'
        )
    return '<div class="flow">' + "".join(parts) + "</div>"


# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("## 🔧 刀具磨损预测与智能诊断")
    st.markdown("**制造智能技术课程设计**")
    st.divider()
    page = st.radio("功能导航", ["总览", "状态识别", "寿命预测", "智能诊断", "历史记录"])
    st.divider()
    ok = backend_ok()
    dot = "dot-ok" if ok else "dot-off"
    text = "后端在线" if ok else "后端离线（需先启动 uvicorn）"
    st.markdown(f'<span class="dot {dot}"></span>{text}', unsafe_allow_html=True)
    st.caption(f"API：`{API}`")


# ============ 总览 ============
if page == "总览":
    health = api("/api/health")
    if health:
        online = '<span class="badge"><span class="dot dot-ok"></span>系统在线</span>'
    else:
        online = '<span class="badge"><span class="dot dot-off"></span>后端未连接</span>'

    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">制造智能技术课程设计</div>
          <h1>CNC 铣削刀具磨损全生命周期预测与智能诊断系统</h1>
          <p>多传感器信号驱动的状态识别、剩余寿命预测与智能诊断决策支持{online}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 关键指标
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("状态准确率", "96.1%", "test 集 386")
    m2.metric("数据样本量", "12,570", "MSM 2570 + AI4I 10000")
    m3.metric("特征维度", "237", "时域 / 频域 / 参数")
    m4.metric("识别类别", "3", "正常 / 异常 / 缺陷")
    m5.metric("功能模块", "4", "识别 / 预测 / 诊断 / 历史")
    m6.metric("技术方向", "3", "优化 / 模式识别 / 知识工程")

    # 核心功能
    header("核心功能", "左侧选择功能页，或点击下方了解各模块")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.container(border=True):
            st.markdown("### 🎯 状态识别")
            st.caption("MSM 多传感器 / 237 维特征")
            st.write("随机森林三分类，识别正常切削、加工异常、刀具缺陷。")
    with c2:
        with st.container(border=True):
            st.markdown("### ⏱ 寿命预测")
            st.caption("AI4I 过程参数回归")
            st.write("回归预测剩余寿命 RUL，临近磨损上限自动预警。")
    with c3:
        with st.container(border=True):
            st.markdown("### 🧠 智能诊断")
            st.caption("知识库 + RAG 检索")
            st.write("异常现象 → 检索可能原因与处置方案。")
    with c4:
        with st.container(border=True):
            st.markdown("### 📋 历史记录")
            st.caption("SQLite 自动入库")
            st.write("所有预测可追溯，支持回看与导出。")

    # 模型性能
    status_m = load_json("models/status_metrics.json")
    rul_m = load_json("models/rul_metrics.json")
    status_p = load_json("models/status_best_params.json")
    rul_p = load_json("models/rul_best_params.json")

    header("模型性能", "测试集评估指标（遗传算法寻优后的随机森林）")

    if status_m:
        acc = status_m["accuracy"]
        mf1 = status_m["macro_f1"]
        wf1 = status_m["weighted_f1"]
        cm = status_m["confusion_matrix"]
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("状态识别准确率", f"{acc:.1%}")
        sm2.metric("宏平均 F1", f"{mf1:.3f}")
        sm3.metric("加权 F1", f"{wf1:.3f}")
        sm4.metric("测试样本", f"{sum(sum(r) for r in cm)}")

        cm_df = pd.DataFrame(
            cm, index=[f"真实·{l}" for l in STATUS_LABELS],
            columns=[f"预测·{l}" for l in STATUS_LABELS],
        )
        left, right = st.columns([3, 2])
        with left:
            st.caption("混淆矩阵（行=真实，列=预测）")
            st.dataframe(cm_df, width="stretch")
        with right:
            st.caption("各类别指标（由混淆矩阵推导）")
            st.dataframe(
                per_class_table(cm).style.format(
                    {"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}
                ),
                width="stretch", hide_index=True,
            )

    if rul_m:
        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("RUL 平均绝对误差", f"{rul_m['mae']:.1f} min")
        rm2.metric("RUL 均方根误差", f"{rul_m['rmse']:.1f} min")
        rm3.metric("RUL R²", f"{rul_m['r2']:.3f}")
        rm4.metric("磨损上限", "253 min")
        st.caption("注：R²≈0.09 说明 AI4I 的 Tool wear 为独立随机变量，过程参数对 RUL 的线性解释力有限，属数据固有局限而非模型缺陷。")

    if status_p and rul_p:
        st.markdown(
            '<div class="panel">'
            f'<span class="k">状态分类 GA 最优参数</span>：<span class="v">n_estimators={status_p["params"]["n_estimators"]}</span> '
            f'<span class="v">max_depth={status_p["params"]["max_depth"]}</span> '
            f'<span class="v">min_samples_leaf={status_p["params"]["min_samples_leaf"]}</span> '
            f'<span class="v">max_features={status_p["params"]["max_features"]}</span><br>'
            f'<span class="k">RUL 回归 GA 最优参数</span>：<span class="v">n_estimators={rul_p["params"]["n_estimators"]}</span> '
            f'<span class="v">max_depth={rul_p["params"]["max_depth"]}</span> '
            f'<span class="v">min_samples_leaf={rul_p["params"]["min_samples_leaf"]}</span> '
            f'<span class="v">max_features={rul_p["params"]["max_features"]}</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    # 系统流程
    header("系统流程", "数据 → 特征 → 模型 → 推理的全链路")
    st.markdown(
        flow_html([
            ("01", "数据采集", "多传感器信号 + 切削参数"),
            ("02", "数据预处理", "缺失填充 / IQR 截断 / 时频域特征提取"),
            ("03", "特征工程", "237 维特征 + Z-score 标准化"),
            ("04", "模型训练", "遗传算法寻优 + 随机森林"),
            ("05", "在线推理", "FastAPI 服务 + Streamlit 看板"),
        ]),
        unsafe_allow_html=True,
    )

    # 技术路线
    header("技术路线", "覆盖课程三大技术方向")
    t1, t2, t3 = st.columns(3)
    with t1:
        with st.container(border=True):
            st.markdown("**⚙️ 智能优化技术**")
            st.write("手写遗传算法（锦标赛选择 + 单点交叉 + 变异 + 精英保留），适应度 = 3 折交叉验证，自动寻优超参数。")
    with t2:
        with st.container(border=True):
            st.markdown("**📊 模式与图像识别**")
            st.write("多传感器信号时频域特征提取 + 随机森林分类器，实现刀具状态自动识别。")
    with t3:
        with st.container(border=True):
            st.markdown("**📚 知识工程**")
            st.write("状态-原因-方案知识库 + TF-IDF 向量检索（RAG），异常时自动推送诊断建议。")

    # 数据来源
    index = load_json("data/processed/index.json")
    header("数据来源", "公开数据集，CC BY 4.0 许可")
    d1, d2 = st.columns(2)
    with d1:
        with st.container(border=True):
            st.markdown("**MSM 多传感器数据集**")
            st.caption("Kaggle · manufuturetoday · CC BY 4.0")
            st.write("2570 标注段 / 237 维特征，标签分布：正常 1928 / 异常 465 / 缺陷 177。")
            st.write("振动、声音、切削电流、MTConnect 控制器数据。")
    with d2:
        with st.container(border=True):
            st.markdown("**AI4I 2020 预测性维护**")
            st.caption("UCI ID 601 · CC BY 4.0")
            st.write("10000 条记录 / 339 失效样本，RUL = max(0, 253 - Tool wear)。")
            st.write("含空气/过程温度、转速、扭矩与连续磨损量，用于 RUL 回归。")

# ============ 状态识别 ============
elif page == "状态识别":
    header("刀具状态识别", "基于 MSM 多传感器特征（237 维）识别正常 / 加工异常 / 刀具缺陷")

    samples = api("/api/samples/status")
    if samples is None:
        st.stop()

    labels = {0: "正常切削", 1: "加工异常", 2: "刀具缺陷"}
    options = [f"{s['id']}（真实：{s['label_name']}）" for s in samples]
    pick = st.selectbox("选择样例（含真实标签，用于验证）", options)
    sample = samples[options.index(pick)]

    with st.expander("查看样例原始特征（方向 / 切削参数 / 信号特征数）", expanded=False):
        st.write("方向：", sample["direction"])
        st.write("切削参数：", sample["params"])
        st.write("信号特征数：", len(sample["signals"]), "个")

    if st.button("预测状态", type="primary"):
        r = api("/api/predict/status", "POST",
                json={"direction": sample["direction"], "params": sample["params"],
                      "signals": sample["signals"]})
        if r:
            c1, c2 = st.columns(2)
            c1.metric("预测结果", r["label_name"], f"置信度 {r['confidence']:.1%}")
            c2.metric("真实标签", sample["label_name"], "✓ 一致" if r["label"] == sample["label"] else "✗ 不一致")
            st.bar_chart(r["probabilities"])

    st.divider()
    status_m = load_json("models/status_metrics.json")
    if status_m:
        with st.expander("模型性能指标（测试集 386 样本）", expanded=False):
            cm = status_m["confusion_matrix"]
            st.write(f"准确率 {status_m['accuracy']:.1%} · 宏平均 F1 {status_m['macro_f1']:.3f}")
            st.dataframe(per_class_table(cm).style.format(
                {"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}
            ), width="stretch", hide_index=True)

# ============ 寿命预测 ============
elif page == "寿命预测":
    header("剩余寿命（RUL）预测", "基于 AI4I 过程参数回归预测刀具剩余寿命（单位：min）")

    rul_m = load_json("models/rul_metrics.json")
    if rul_m:
        st.caption(
            f"回归指标：MAE {rul_m['mae']:.1f} min / RMSE {rul_m['rmse']:.1f} min / "
            f"R² {rul_m['r2']:.3f}（过程参数对磨损的解释力有限，属数据固有局限）"
        )

    mode = st.radio("输入方式", ["使用样例", "手动输入"], horizontal=True)

    if mode == "使用样例":
        samples = api("/api/samples/rul")
        if samples is None:
            st.stop()
        options = [f"{s['id']}（真实 RUL={s['rul']:.0f} min）" for s in samples]
        pick = st.selectbox("选择样例", options)
        sample = samples[options.index(pick)]
        type_ = sample["type"]
        air = sample["air_temperature"]
        proc = sample["process_temperature"]
        speed = sample["rotational_speed"]
        torque = sample["torque"]
        st.info(f"Type={type_} | 空气温度 {air:.1f} K | 过程温度 {proc:.1f} K | 转速 {speed:.0f} rpm | 扭矩 {torque:.1f} Nm")
    else:
        type_ = st.selectbox("产品类型 Type", ["L", "M", "H"])
        air = st.number_input("空气温度 Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
        proc = st.number_input("过程温度 Process temperature [K]", 305.0, 315.0, 310.0, 0.1)
        speed = st.number_input("转速 Rotational speed [rpm]", 1200.0, 2900.0, 1500.0, 10.0)
        torque = st.number_input("扭矩 Torque [Nm]", 3.0, 70.0, 40.0, 0.5)

    if st.button("预测 RUL", type="primary"):
        r = api("/api/predict/rul", "POST",
                json={"type": type_, "air_temperature": air, "process_temperature": proc,
                      "rotational_speed": speed, "torque": torque})
        if r:
            st.metric("预测 RUL", f"{r['rul']:.1f} min")
            if r.get("warning"):
                st.warning(r["warning"])
            st.progress(min(1.0, r["rul"] / 253.0), text="剩余寿命占比")

# ============ 智能诊断 ============
elif page == "智能诊断":
    header("智能诊断（RAG 知识库检索）", "输入异常现象描述，检索知识库返回原因与处置方案")

    kb = api("/api/diagnosis/kb")
    if kb:
        st.caption(f"知识库共 {len(kb)} 条（状态 / 现象关键词 / 原因 / 处置方案）")

    query = st.text_input("现象描述", placeholder="例如：主轴负载异常升高 / 刀具崩刃 / 异常噪声")
    top_k = st.slider("返回条数 Top-K", 1, 5, 3)

    if st.button("诊断", type="primary") or query:
        if query.strip():
            r = api("/api/diagnosis/query", "POST", json={"query": query, "top_k": top_k})
            if r:
                if not r["results"]:
                    st.info("未检索到相关条目，请换一种描述。")
                for item in r["results"]:
                    with st.container(border=True):
                        st.markdown(f"**{item['status']}**（相似度 {item['similarity']:.3f}），{item['keywords']}")
                        st.markdown(f"**可能原因**：{item['cause']}")
                        st.markdown(f"**处置方案**：{item['solution']}")
        else:
            st.info("请输入现象描述。")

    with st.expander("查看知识库全部条目", expanded=False):
        if kb:
            for d in kb:
                st.markdown(f"- **{d['status']}**：{d['keywords']} → {d['solution']}")

# ============ 历史记录 ============
elif page == "历史记录":
    header("预测历史记录", "所有预测 / 诊断自动入库，可追溯")

    if st.button("刷新"):
        st.rerun()
    rows = api("/api/history?limit=100")
    if rows is None:
        st.stop()
    if not rows:
        st.info("暂无记录。")
    else:
        n1, n2, n3 = st.columns(3)
        n1.metric("总记录数", len(rows))
        kinds = {}
        for r in rows:
            kinds[r.get("kind", "未知")] = kinds.get(r.get("kind", "未知"), 0) + 1
        n2.metric("类型数", len(kinds))
        n3.metric("最近一次", rows[0]["ts"][:19] if rows[0].get("ts") else "-")
        df = pd.DataFrame(rows)
        df = df.rename(columns={"ts": "时间", "kind": "类型", "input_summary": "输入摘要",
                                "result": "结果", "confidence": "置信度"})
        st.dataframe(df, width="stretch")
