#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.regime.rss_vote_utils import rss_vote_confidence, rss_vote_state


@dataclass
class RssCacheSnapshot:
    collected_at: datetime
    source_confidence: float


def _parse_cache_time(path: Path, payload: dict) -> Optional[datetime]:
    raw = payload.get("collected_at")
    if raw:
        try:
            return datetime.fromisoformat(str(raw))
        except ValueError:
            pass

    parts = path.stem.split("_")
    if len(parts) >= 3:
        try:
            return datetime.strptime("_".join(parts[-2:]), "%Y%m%d_%H")
        except ValueError:
            return None
    return None


def _load_rss_cache_snapshots(cache_dir: Path) -> List[RssCacheSnapshot]:
    snapshots: List[RssCacheSnapshot] = []
    for path in sorted(cache_dir.glob("rss_MARKET_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        collected_at = _parse_cache_time(path, payload)
        if collected_at is None:
            continue
        source_confidence = float(payload.get("f6_sentiment_confidence", 0.7) or 0.7)
        snapshots.append(
            RssCacheSnapshot(
                collected_at=collected_at,
                source_confidence=source_confidence,
            )
        )
    snapshots.sort(key=lambda item: item.collected_at)
    return snapshots


def _source_confidence_for(ts_ms: int, snapshots: List[RssCacheSnapshot]) -> float:
    if not snapshots:
        return 0.7
    target = datetime.fromtimestamp(ts_ms / 1000)
    times = [item.collected_at for item in snapshots]
    idx = bisect_right(times, target) - 1
    if idx < 0:
        return snapshots[0].source_confidence
    return snapshots[idx].source_confidence


def _resolve_cutoff_ms(cur: sqlite3.Cursor, snapshots: List[RssCacheSnapshot], hours: int) -> int:
    if hours <= 0:
        return 0

    reference_ts_ms: List[int] = []

    cur.execute("SELECT MAX(ts_ms) FROM regime_history WHERE rss_sentiment IS NOT NULL")
    row = cur.fetchone()
    if row and row[0] is not None:
        reference_ts_ms.append(int(row[0]))

    if snapshots:
        reference_ts_ms.append(int(max(item.collected_at for item in snapshots).timestamp() * 1000))

    if not reference_ts_ms:
        return 0

    window_ms = int(timedelta(hours=hours).total_seconds() * 1000)
    return max(0, max(reference_ts_ms) - window_ms)


def backfill_regime_history_rss(db_path: Path, cache_dir: Path, hours: int) -> dict:
    snapshots = _load_rss_cache_snapshots(cache_dir)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cutoff_ms = _resolve_cutoff_ms(cur, snapshots, hours)
    cur.execute(
        """
        SELECT id, ts_ms, rss_sentiment, rss_state, rss_confidence
        FROM regime_history
        WHERE ts_ms >= ?
          AND rss_sentiment IS NOT NULL
        ORDER BY ts_ms ASC
        """,
        (cutoff_ms,),
    )
    rows = cur.fetchall()

    updated = 0
    changed = []
    for row_id, ts_ms, rss_sentiment, old_state, old_confidence in rows:
        source_confidence = _source_confidence_for(int(ts_ms), snapshots)
        new_state = rss_vote_state(float(rss_sentiment or 0.0))
        new_confidence = rss_vote_confidence(float(rss_sentiment or 0.0), source_confidence)
        if old_state == new_state and abs(float(old_confidence or 0.0) - new_confidence) < 1e-9:
            continue
        cur.execute(
            """
            UPDATE regime_history
            SET rss_state = ?, rss_confidence = ?
            WHERE id = ?
            """,
            (new_state, float(new_confidence), row_id),
        )
        updated += 1
        changed.append(
            {
                "id": row_id,
                "ts_ms": ts_ms,
                "old_state": old_state,
                "new_state": new_state,
                "old_confidence": float(old_confidence or 0.0),
                "new_confidence": float(new_confidence),
                "source_confidence": float(source_confidence),
            }
        )

    conn.commit()
    conn.close()
    return {
        "db_path": str(db_path),
        "cache_dir": str(cache_dir),
        "hours": hours,
        "rows_scanned": len(rows),
        "rows_updated": updated,
        "samples": changed[-5:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill RSS confidence in regime_history.db")
    parser.add_argument("--db-path", default="reports/regime_history.db")
    parser.add_argument("--cache-dir", default="data/sentiment_cache")
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()

    result = backfill_regime_history_rss(
        db_path=Path(args.db_path),
        cache_dir=Path(args.cache_dir),
        hours=args.hours,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
