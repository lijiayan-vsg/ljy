# -*- coding: utf-8 -*-
"""SQLite 预测历史记录（每次操作独立连接，避免线程间共享连接）。

针对 Windows 下偶发的文件锁（Defender 扫描导致的 SQLITE_READONLY/SQLITE_BUSY），
连接加超时，写入做有限重试，并启用 WAL 降低并发写锁竞争。
"""
import sqlite3
import time
from datetime import datetime

from . import config


def _conn():
    # timeout 让并发写等待而非立即报错；WAL 降低读写锁竞争
    c = sqlite3.connect(config.DB_PATH, timeout=5.0)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def _retry(fn, attempts=3, delay=0.1):
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            last = e
            time.sleep(delay)
    raise last


def init_db():
    def _do():
        c = _conn()
        try:
            c.execute(
                """CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    input_summary TEXT,
                    result TEXT,
                    confidence REAL
                )"""
            )
            c.commit()
        finally:
            c.close()

    _retry(_do)


def add_prediction(kind: str, input_summary: str, result: str, confidence=None):
    def _do():
        c = _conn()
        try:
            c.execute(
                "INSERT INTO predictions (ts, kind, input_summary, result, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), kind, input_summary, result, confidence),
            )
            c.commit()
        finally:
            c.close()

    _retry(_do)


def list_predictions(limit: int = 50):
    c = _conn()
    rows = c.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]
