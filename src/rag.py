# -*- coding: utf-8 -*-
"""
知识库检索增强（RAG）—— 知识工程模块。

用 sklearn 的 TF-IDF 向量化知识库文档，检索时以余弦相似度召回 Top-K 最相关条目，
返回「状态 → 可能原因 → 处置方案」，实现异常检测后的智能诊断决策支持。

实现说明：不引入 chromadb 等重依赖，采用轻量 TF-IDF + 余弦相似度完成向量检索；
KnowledgeBase 已抽象出独立接口，后续可无缝替换为 ChromaDB 持久化向量库。

运行（在项目根目录 d:/ljy 下）：
    python -m src.rag                          # 演示多个查询
    python -m src.rag --query "主轴振动异常" --top-k 3
"""
import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import paths


class KnowledgeBase:
    """轻量向量知识库：TF-IDF 向量化 + 余弦相似度检索。

    docs 结构：[{id, status, keywords, cause, solution}, ...]
    """

    def __init__(self, docs):
        self.docs = docs
        # 中文无空格分词：用字符 n-gram（char_wb，2~3 元）向量化，
        # 无需 jieba 等分词依赖即可对未分词的查询文本召回匹配。
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
        self._build()

    def _text(self, d):
        return " ".join([d.get("status", ""), d.get("keywords", ""), d.get("cause", "")])

    def _build(self):
        corpus = [self._text(d) for d in self.docs]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def query(self, text, top_k=3):
        """返回 [(文档, 相似度), ...]，按相似度降序，过滤掉零相似度条目。"""
        q = self.vectorizer.transform([text])
        sim = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(sim)[::-1][:top_k]
        return [(self.docs[i], float(sim[i])) for i in order if sim[i] > 0]


def load_knowledge_base():
    with open(paths.KNOWLEDGE_BASE, encoding="utf-8") as f:
        data = json.load(f)
    return data["documents"]


def main(query=None, top_k=3):
    docs = load_knowledge_base()
    kb = KnowledgeBase(docs)
    queries = [query] if query else [
        "主轴负载异常升高",
        "刀具崩刃 表面粗糙",
        "切削时发出异常噪声",
        "切削过程平稳正常",
    ]
    for q in queries:
        print(f"\n【查询】{q}")
        results = kb.query(q, top_k=top_k)
        if not results:
            print("  （未检索到相关条目）")
            continue
        for d, s in results:
            print(f"  相似度={s:.3f} | 状态={d['status']} | 关键词={d['keywords']}")
            print(f"    原因: {d['cause']}")
            print(f"    方案: {d['solution']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAG 知识库检索演示")
    ap.add_argument("--query", default=None, help="单条查询文本；缺省则演示多条查询")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()
    main(args.query, args.top_k)
