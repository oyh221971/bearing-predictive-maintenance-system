"""SQLite 存储层：监测样本记录与告警记录，线程安全。"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from app.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    scenario      TEXT NOT NULL,
    fault_type    TEXT NOT NULL,
    confidence    REAL NOT NULL,
    severity      REAL NOT NULL,
    rul_hours     REAL NOT NULL,
    health_index  REAL NOT NULL,
    rms           REAL NOT NULL,
    kurtosis      REAL NOT NULL,
    crest         REAL NOT NULL,
    temperature   REAL NOT NULL,
    probs         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    level   TEXT NOT NULL,
    message TEXT NOT NULL,
    active  INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts);
CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts(active);
"""


class Database:
    """轻量 SQLite 封装：所有写操作串行化，支持多线程（FastAPI/监测线程）。"""

    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------- 样本 ----------
    def insert_sample(self, *, scenario: str, fault_type: str, confidence: float,
                      severity: float, rul_hours: float, health_index: float,
                      rms: float, kurtosis: float, crest: float,
                      temperature: float, probs: dict) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO samples
                   (ts, scenario, fault_type, confidence, severity, rul_hours,
                    health_index, rms, kurtosis, crest, temperature, probs)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (time.time(), scenario, fault_type, confidence, severity,
                 rul_hours, health_index, rms, kurtosis, crest, temperature,
                 json.dumps(probs, ensure_ascii=False)))
            self._conn.commit()
            return int(cur.lastrowid)

    def recent_samples(self, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM samples ORDER BY id DESC LIMIT ? OFFSET ?""",
                (max(1, min(limit, 2000)), max(0, offset))).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["probs"] = json.loads(d["probs"])
            out.append(d)
        return out

    # ---------- 告警 ----------
    def sync_alerts(self, desired: list[tuple[str, str]]) -> None:
        """将活跃告警同步为 desired（(message, level) 列表）。

        新出现的消息插入；不再出现的消息自动解除（active=0）。
        """
        with self._lock:
            desired_msgs = {m for m, _ in desired}
            for row in self._conn.execute(
                    "SELECT id, message FROM alerts WHERE active=1").fetchall():
                if row["message"] not in desired_msgs:
                    self._conn.execute(
                        "UPDATE alerts SET active=0 WHERE id=?", (row["id"],))
            for message, level in desired:
                exists = self._conn.execute(
                    "SELECT 1 FROM alerts WHERE active=1 AND message=?",
                    (message,)).fetchone()
                if not exists:
                    self._conn.execute(
                        "INSERT INTO alerts (ts, level, message, active) "
                        "VALUES (?,?,?,1)", (time.time(), level, message))
            self._conn.commit()

    def active_alerts(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM alerts WHERE active=1 ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_alerts(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, 500)),)).fetchall()
        return [dict(r) for r in rows]

    def ack_alert(self, alert_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE alerts SET active=0 WHERE id=? AND active=1",
                (alert_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ---------- 统计 ----------
    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) AS c FROM samples").fetchone()["c"]
            by_fault = {r["fault_type"]: r["c"] for r in self._conn.execute(
                "SELECT fault_type, COUNT(*) AS c FROM samples "
                "GROUP BY fault_type").fetchall()}
            by_scenario = {r["scenario"]: r["c"] for r in self._conn.execute(
                "SELECT scenario, COUNT(*) AS c FROM samples "
                "GROUP BY scenario").fetchall()}
            last = self._conn.execute(
                "SELECT rul_hours, health_index, severity, temperature, ts "
                "FROM samples ORDER BY id DESC LIMIT 1").fetchone()
            active_alerts = self._conn.execute(
                "SELECT COUNT(*) AS c FROM alerts WHERE active=1").fetchone()["c"]
            total_alerts = self._conn.execute(
                "SELECT COUNT(*) AS c FROM alerts").fetchone()["c"]
        return {
            "total_samples": total,
            "by_fault_type": by_fault,
            "by_scenario": by_scenario,
            "active_alerts": active_alerts,
            "total_alerts": total_alerts,
            "last": dict(last) if last else None,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
