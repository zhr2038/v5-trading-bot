import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "web_dashboard.py"
MONITOR_V2_JS_PATH = REPO_ROOT / "web" / "static" / "js" / "monitor_v2.js"


def load_web_dashboard_module():
    name = f"web_dashboard_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _find_headless_browser() -> str | None:
    candidates = [
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return None


def test_index_renders_monitor_template():
    module = load_web_dashboard_module()
    client = module.app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'id="update-time"' in body
    assert 'id="health-content"' in body
    assert 'id="vote-history"' in body
    assert 'id="history-tooltip"' in body
    assert 'id="position-kline-chart"' in body
    assert 'id="position-kline-symbols"' in body
    assert 'id="position-kline-timeframes"' in body
    assert 'id="ml-impact-headline"' in body
    assert 'id="ml-impact-subtitle"' in body
    assert 'id="ml-impact-content"' in body
    assert 'src="/static/js/monitor_v2.js?v=' in body
    assert 'src="/static/js/ml_status_panel.js?v=' in body


def test_monitor_v2_static_script_contains_expected_entrypoints():
    body = MONITOR_V2_JS_PATH.read_text(encoding="utf-8")

    assert "renderHmmProbRows" in body
    assert "renderDerivedVoteRows" in body
    assert "renderVoteHistory" in body
    assert "renderMlSignalCard" in body
    assert "showHistoryTooltip" in body
    assert "showHmmProbs:true" in body
    assert "showStateBars:true" in body
    assert "showSummary:true" in body
    assert "renderShadowMlPanel" in body
    assert "/api/shadow_ml_overlay" in body
    assert "buildCandlestickSvg" in body
    assert "syncPositionSpotlight" in body
    assert "/api/position_kline" in body
    assert "position-kline-timeframes" in body
    assert "metrics.conversion_rate??metrics.last_conversion_rate??null" in body
    assert "account?.drawdown_pct??metrics.dd_pct??metrics.last_dd_pct??null" in body
    assert "item.display_score ?? item.score ?? 0" in body
    assert "item.raw_score ?? displayScore" in body
    assert "原始 ${fmtNum(rawScore, 3)}" in body
    assert "浮盈亏 / 收益率" in body
    assert "fmtUsd(pnlValue)" in body
    assert "p.pnl??p.pnl_value||0" not in body
    assert "p.pnlPercent??p.pnl_pct||0" not in body
    assert "loadAll();" in body


def test_monitor_v2_static_script_executes_in_headless_browser(tmp_path):
    browser = _find_headless_browser()
    if not browser:
        pytest.skip("No Chromium-based browser available for JS syntax check")

    script_path = tmp_path / "monitor_v2.js"
    script_path.write_text(MONITOR_V2_JS_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    harness_path = tmp_path / "monitor_v2_harness.html"
    harness_path.write_text(
        f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>monitor_v2 harness</title></head>
<body>
<div id="update-time"></div>
<script>
window.fetch = async () => ({{ json: async () => ({{}}) }});
window.setInterval = () => 0;
</script>
<script src="{script_path.as_uri()}"></script>
</body>
</html>
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--enable-logging=stderr",
            "--v=1",
            "--virtual-time-budget=3000",
            "--dump-dom",
            harness_path.as_uri(),
        ],
        capture_output=True,
        text=False,
        timeout=30,
        check=False,
    )

    stdout = (result.stdout or b"").decode("utf-8", errors="ignore")
    stderr = (result.stderr or b"").decode("utf-8", errors="ignore")
    combined_output = "\n".join(part for part in [stdout, stderr] if part)
    assert "SyntaxError" not in combined_output
    assert "Uncaught" not in combined_output
    assert "最近刷新" in stdout


def test_timer_endpoints_degrade_without_systemctl(monkeypatch):
    module = load_web_dashboard_module()
    client = module.app.test_client()
    monkeypatch.setattr(module, "SYSTEMCTL_BIN", None)

    status_response = client.get("/api/status")
    timer_response = client.get("/api/timer")
    timers_response = client.get("/api/timers")

    assert status_response.status_code == 200
    assert timer_response.status_code == 200
    assert timers_response.status_code == 200

    status_payload = status_response.get_json()
    timer_payload = timer_response.get_json()
    timers_payload = timers_response.get_json()

    assert status_payload["timer_active"] is False
    assert "systemctl" in status_payload["timer_error"]
    assert timer_payload["next_run"] is None
    assert "systemctl" in timer_payload["error"]
    assert timers_payload["timers"]
    assert all("error" in timer for timer in timers_payload["timers"])


def test_dashboard_api_uses_expected_payload_shapes(monkeypatch):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    positions_rows = [{
        "symbol": "BTC",
        "qty": 0.1,
        "avg_px": 50000.0,
        "last_price": 55000.0,
        "value_usdt": 5500.0,
        "price": 55000.0,
        "value": 5500.0,
        "quantity": 0.1,
        "pnl_pct": 0.1,
    }]
    account_payload = {
        "cash_usdt": 100.0,
        "positions_value_usdt": 5500.0,
        "total_equity_usdt": 5600.0,
        "total_pnl_pct": 0.25,
        "drawdown_pct": 0.05,
        "realized_pnl": 12.5,
        "total_trades": 7,
        "last_update": "2026-03-08 21:00:00",
    }

    monkeypatch.setattr(module, "api_positions", lambda: module.jsonify({"positions": positions_rows}))
    monkeypatch.setattr(module, "api_account", lambda: module.jsonify(account_payload))
    monkeypatch.setattr(module, "api_trades", lambda: module.jsonify({"trades": [{
        "time": "2026-03-08 20:00:00",
        "symbol": "BTC-USDT",
        "side": "buy",
        "amount": 100.0,
        "fee": -0.1,
    }]}))
    monkeypatch.setattr(module, "api_scores", lambda: module.jsonify({"scores": [{
        "symbol": "BTC-USDT",
        "score": 0.9,
    }]}))
    monkeypatch.setattr(module, "api_status", lambda: module.jsonify({
        "timer_active": False,
        "dry_run": True,
        "timer_error": "systemctl is not available",
    }))
    monkeypatch.setattr(module, "api_equity_history", lambda: module.jsonify([{
        "timestamp": "2026-03-08T21:00:00",
        "value": 5600.0,
    }]))
    monkeypatch.setattr(module, "api_market_state", lambda: module.jsonify({
        "state": "TRENDING",
        "position_multiplier": 1.2,
    }))
    monkeypatch.setattr(module, "api_timers", lambda: module.jsonify({
        "timers": [{"name": "v5-prod.user.timer", "active": False, "enabled": False, "next_run": None}],
        "next_run": None,
    }))
    monkeypatch.setattr(module, "api_cost_calibration", lambda: module.jsonify({"status": "ok"}))
    monkeypatch.setattr(module, "api_ic_diagnostics", lambda: module.jsonify({"status": "ok"}))
    monkeypatch.setattr(module, "api_ml_training", lambda: module.jsonify({"status": "idle"}))
    monkeypatch.setattr(module, "api_reflection_reports", lambda: module.jsonify({"reports": []}))

    response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["account"]["totalEquity"] == 5600.0
    assert payload["account"]["totalPnlPercent"] == 0.25
    assert payload["account"]["maxDrawdown"] == 0.05
    assert payload["positions"][0]["symbol"] == "BTC"
    assert payload["positions"][0]["pnlPercent"] == 0.1
    assert payload["trades"][0]["symbol"] == "BTC/USDT"
    assert payload["alphaScores"][0]["symbol"] == "BTC/USDT"
    assert payload["systemStatus"]["errors"] == ["systemctl is not available"]


def test_dashboard_api_keeps_sub_one_percent_pnl_as_ratio(monkeypatch):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    monkeypatch.setattr(module, "api_positions", lambda: module.jsonify({"positions": [{
        "symbol": "OKB",
        "qty": 0.67,
        "avg_px": 97.197842,
        "last_price": 96.67,
        "value_usdt": 65.2311,
        "pnl_value": -0.3562,
        "pnl_pct": -0.0054,
    }]}))
    monkeypatch.setattr(module, "api_account", lambda: module.jsonify({
        "cash_usdt": 100.0,
        "positions_value_usdt": 65.2311,
        "total_equity_usdt": 165.2311,
        "total_pnl_pct": 0.005,
        "drawdown_pct": 0.0075,
        "realized_pnl": 1.25,
        "total_trades": 3,
        "last_update": "2026-03-09 21:00:00",
    }))
    monkeypatch.setattr(module, "api_trades", lambda: module.jsonify({"trades": []}))
    monkeypatch.setattr(module, "api_scores", lambda: module.jsonify({"scores": []}))
    monkeypatch.setattr(module, "api_status", lambda: module.jsonify({}))
    monkeypatch.setattr(module, "api_equity_history", lambda: module.jsonify([]))
    monkeypatch.setattr(module, "api_market_state", lambda: module.jsonify({}))
    monkeypatch.setattr(module, "api_timers", lambda: module.jsonify({"timers": [], "next_run": None}))
    monkeypatch.setattr(module, "api_cost_calibration", lambda: module.jsonify({"status": "ok"}))
    monkeypatch.setattr(module, "api_ic_diagnostics", lambda: module.jsonify({"status": "ok"}))
    monkeypatch.setattr(module, "api_ml_training", lambda: module.jsonify({"status": "idle"}))
    monkeypatch.setattr(module, "api_reflection_reports", lambda: module.jsonify({"reports": []}))

    response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["account"]["totalPnlPercent"] == 0.005
    assert payload["account"]["maxDrawdown"] == 0.0075
    assert payload["positions"][0]["pnlPercent"] == -0.0054


def test_position_kline_api_returns_expected_shape(monkeypatch):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    series = module.MarketSeries(
        symbol="BTC/USDT",
        timeframe="1h",
        ts=[1710201600000, 1710205200000],
        open=[100.0, 101.0],
        high=[102.0, 105.0],
        low=[99.5, 100.5],
        close=[101.0, 103.0],
        volume=[10.0, 12.0],
    )
    monkeypatch.setattr(module, "_load_position_market_series", lambda symbol, timeframe, limit: (series, "cache"))

    response = client.get("/api/position_kline?symbol=BTC&timeframe=1h&limit=2")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["symbol"] == "BTC/USDT"
    assert payload["timeframe"] == "1h"
    assert payload["source"] == "cache"
    assert len(payload["candles"]) == 2
    assert payload["candles"][0]["open"] == 100.0
    assert payload["summary"]["bars"] == 2
    assert payload["summary"]["high"] == 105.0
    assert payload["summary"]["low"] == 99.5
    assert payload["summary"]["volume"] == 22.0
    assert payload["summary"]["change_pct"] == pytest.approx(0.03)


def test_position_kline_api_normalizes_second_timestamps(monkeypatch):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    series = module.MarketSeries(
        symbol="BTC/USDT",
        timeframe="1h",
        ts=[1710201600, 1710205200],
        open=[100.0, 101.0],
        high=[102.0, 105.0],
        low=[99.5, 100.5],
        close=[101.0, 103.0],
        volume=[10.0, 12.0],
    )
    monkeypatch.setattr(module, "_load_position_market_series", lambda symbol, timeframe, limit: (series, "cache"))

    response = client.get("/api/position_kline?symbol=BTC&timeframe=1h")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["candles"][0]["ts"] == 1710201600000
    assert payload["candles"][0]["time"] == "2024-03-12 00:00"


def test_api_scores_exposes_display_score_rank_and_raw_strength(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    reports_dir = tmp_path / "reports"
    runs_dir = reports_dir / "runs"
    previous_run = runs_dir / "20260311_00"
    current_run = runs_dir / "20260311_01"
    previous_run.mkdir(parents=True, exist_ok=True)
    current_run.mkdir(parents=True, exist_ok=True)

    (previous_run / "decision_audit.json").write_text(
        json.dumps(
            {
                "regime": "SIDEWAYS",
                "top_scores": [
                    {"symbol": "HYPE/USDT", "score": 0.91, "display_score": 0.91, "raw_score": 1.82, "rank": 1},
                    {"symbol": "FLOW/USDT", "score": 0.88, "display_score": 0.88, "raw_score": 1.90, "rank": 2},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (current_run / "decision_audit.json").write_text(
        json.dumps(
            {
                "regime": "SIDEWAYS",
                "top_scores": [
                    {"symbol": "FLOW/USDT", "score": 0.9987, "display_score": 0.9987, "raw_score": 3.7228, "rank": 1},
                    {"symbol": "HYPE/USDT", "score": 0.9521, "display_score": 0.9521, "raw_score": 0.9521, "rank": 2},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.utime(previous_run, (1, 1))
    os.utime(current_run, (2, 2))
    monkeypatch.setattr(module, "REPORTS_DIR", reports_dir)

    response = client.get("/api/scores")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["regime"] == "SIDEWAYS"
    first = payload["scores"][0]
    assert first["symbol"] == "FLOW/USDT"
    assert first["rank"] == 1
    assert first["previous_rank"] == 2
    assert first["rank_change"] == 1
    assert first["score"] == 0.9987
    assert first["display_score"] == 0.9987
    assert first["raw_score"] == 3.7228
    assert first["raw_score_change"] == pytest.approx(1.8228)


def test_api_scores_backfills_display_score_for_legacy_raw_values(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    reports_dir = tmp_path / "reports"
    runs_dir = reports_dir / "runs"
    current_run = runs_dir / "20260311_01"
    current_run.mkdir(parents=True, exist_ok=True)
    (current_run / "decision_audit.json").write_text(
        json.dumps(
            {
                "regime": "SIDEWAYS",
                "top_scores": [
                    {"symbol": "FLOW/USDT", "score": 3.7228},
                    {"symbol": "HYPE/USDT", "score": 0.9521},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.utime(current_run, (2, 2))
    monkeypatch.setattr(module, "REPORTS_DIR", reports_dir)

    response = client.get("/api/scores")

    assert response.status_code == 200
    payload = response.get_json()
    first = payload["scores"][0]
    assert first["symbol"] == "FLOW/USDT"
    assert first["raw_score"] == 3.7228
    assert 0.99 < first["display_score"] < 1.0
    assert first["score"] == first["display_score"]


def test_api_scores_skips_latest_failed_run(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    reports_dir = tmp_path / "reports"
    runs_dir = reports_dir / "runs"
    valid_run = runs_dir / "20260311_00"
    failed_run = runs_dir / "20260311_01"
    valid_run.mkdir(parents=True, exist_ok=True)
    failed_run.mkdir(parents=True, exist_ok=True)

    (valid_run / "decision_audit.json").write_text(
        json.dumps(
            {
                "regime": "SIDEWAYS",
                "top_scores": [
                    {"symbol": "FLOW/USDT", "score": 0.88, "display_score": 0.88, "raw_score": 1.90, "rank": 1},
                    {"symbol": "HYPE/USDT", "score": 0.86, "display_score": 0.86, "raw_score": 1.70, "rank": 2},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (failed_run / "decision_audit.json").write_text(
        json.dumps(
            {
                "regime": "Unknown",
                "top_scores": [],
                "counts": {"universe": 0, "scored": 0},
                "notes": ["No market data returned from provider"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.utime(valid_run, (1, 1))
    os.utime(failed_run, (2, 2))
    monkeypatch.setattr(module, "REPORTS_DIR", reports_dir)

    response = client.get("/api/scores")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["current_run"] == "20260311_00"
    assert payload["regime"] == "SIDEWAYS"
    assert payload["scores"][0]["symbol"] == "FLOW/USDT"


def test_api_scores_falls_back_to_alpha_snapshot_when_runs_empty(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    reports_dir = tmp_path / "reports"
    runs_dir = reports_dir / "runs"
    failed_run = runs_dir / "20260311_01"
    failed_run.mkdir(parents=True, exist_ok=True)
    (failed_run / "decision_audit.json").write_text(
        json.dumps(
            {
                "regime": "Unknown",
                "top_scores": [],
                "counts": {"universe": 0, "scored": 0},
                "notes": ["No market data returned from provider"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (reports_dir / "alpha_snapshot.json").write_text(
        json.dumps(
            {
                "scores": {
                    "SOL/USDT": 0.95,
                    "BTC/USDT": 0.75,
                    "ETH/USDT": -0.10,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (reports_dir / "regime.json").write_text(
        json.dumps({"state": "Trending", "multiplier": 1.2, "votes": {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    os.utime(failed_run, (2, 2))
    monkeypatch.setattr(module, "REPORTS_DIR", reports_dir)

    response = client.get("/api/scores")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["current_run"] == "alpha_snapshot"
    assert payload["regime"] == "Trending"
    assert [item["symbol"] for item in payload["scores"][:2]] == ["SOL/USDT", "BTC/USDT"]


def test_api_decision_audit_exposes_ml_signal_overview(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    reports_dir = tmp_path / "reports"
    runs_dir = reports_dir / "runs"
    current_run = runs_dir / "20260312_01"
    current_run.mkdir(parents=True, exist_ok=True)
    (current_run / "decision_audit.json").write_text(
        json.dumps(
            {
                "run_id": "20260312_01",
                "regime": "TRENDING",
                "counts": {"selected": 2, "orders_rebalance": 1, "orders_exit": 0},
                "top_scores": [
                    {"symbol": "BTC/USDT", "score": 0.93, "display_score": 0.93, "raw_score": 1.44, "rank": 1},
                    {"symbol": "ETH/USDT", "score": 0.84, "display_score": 0.84, "raw_score": 0.88, "rank": 2},
                ],
                "targets_pre_risk": {"BTC/USDT": 0.35, "ETH/USDT": 0.25},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (reports_dir / "ml_runtime_status.json").write_text(
        json.dumps(
            {
                "configured_enabled": True,
                "promotion_passed": True,
                "used_in_latest_snapshot": True,
                "prediction_count": 3,
                "ml_weight": 0.2,
                "configured_ml_weight": 0.2,
                "effective_ml_weight": 0.08,
                "overlay_mode": "downweighted",
                "online_control_reason": "rolling_24h_negative",
                "reason": "ok",
                "ts": "2026-03-12T02:01:08Z",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (reports_dir / "alpha_snapshot.json").write_text(
        json.dumps(
            {
                "raw_factors": {
                    "BTC/USDT": {"ml_pred_raw": 0.061, "ml_overlay_score": 0.52, "ml_base_score": 0.78},
                    "ETH/USDT": {"ml_pred_raw": -0.018, "ml_overlay_score": -0.24, "ml_base_score": 0.91},
                    "SOL/USDT": {"ml_pred_raw": 0.074, "ml_overlay_score": 0.68, "ml_base_score": 0.55},
                },
                "z_factors": {
                    "BTC/USDT": {"ml_pred_zscore": 0.82, "ml_overlay_score": 0.52},
                    "ETH/USDT": {"ml_pred_zscore": -0.31, "ml_overlay_score": -0.24},
                    "SOL/USDT": {"ml_pred_zscore": 1.12, "ml_overlay_score": 0.68},
                },
                "base_scores": {
                    "BTC/USDT": 0.78,
                    "ETH/USDT": 0.91,
                    "SOL/USDT": 0.55,
                },
                "scores": {
                    "BTC/USDT": 0.93,
                    "ETH/USDT": 0.84,
                    "SOL/USDT": 0.65,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (reports_dir / "ml_overlay_impact.json").write_text(
        json.dumps(
            {
                "last_step": {
                    "top_n": 3,
                    "delta_bps": 14.2,
                    "status": "positive",
                },
                "rolling_24h": {
                    "points": 6,
                    "topn_delta_mean_bps": 8.4,
                    "status": "positive",
                },
                "rolling_48h": {
                    "points": 12,
                    "topn_delta_mean_bps": -3.1,
                    "status": "mixed",
                },
                "overlay_mode": "downweighted",
                "online_control_reason": "rolling_24h_negative",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.utime(current_run, (2, 2))
    monkeypatch.setattr(module, "REPORTS_DIR", reports_dir)

    response = client.get("/api/decision_audit")

    assert response.status_code == 200
    payload = response.get_json()
    ml = payload["ml_signal_overview"]
    assert ml["configured_enabled"] is True
    assert ml["promoted"] is True
    assert ml["live_active"] is True
    assert ml["prediction_count"] == 3
    assert ml["ml_weight"] == 0.2
    assert ml["configured_ml_weight"] == 0.2
    assert ml["effective_ml_weight"] == 0.08
    assert ml["overlay_mode"] == "downweighted"
    assert ml["online_control_reason"] == "rolling_24h_negative"
    assert ml["impact_status"] == "positive"
    assert ml["last_step"]["delta_bps"] == 14.2
    assert ml["rolling_24h"]["topn_delta_mean_bps"] == 8.4
    assert ml["rolling_48h"]["topn_delta_mean_bps"] == -3.1
    assert [item["symbol"] for item in ml["top_contributors"]] == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    assert ml["top_contributors"][0]["ml_zscore"] == 0.82
    assert ml["top_promoted"][0]["symbol"] == "BTC/USDT"
    assert ml["top_suppressed"][0]["symbol"] == "ETH/USDT"


def test_api_decision_audit_prefers_current_run_embedded_strategy_signals(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    reports_dir = tmp_path / "reports"
    runs_dir = reports_dir / "runs"
    current_run = runs_dir / "20260313_14"
    stale_run = runs_dir / "20260313_13"
    current_run.mkdir(parents=True, exist_ok=True)
    stale_run.mkdir(parents=True, exist_ok=True)

    (current_run / "decision_audit.json").write_text(
        json.dumps(
            {
                "run_id": "20260313_14",
                "regime": "SIDEWAYS",
                "counts": {"selected": 1, "orders_rebalance": 0, "orders_exit": 0},
                "strategy_signals": [
                    {
                        "strategy": "MeanReversion",
                        "allocation": 0.25,
                        "total_signals": 1,
                        "buy_signals": 0,
                        "sell_signals": 1,
                        "signals": [{"symbol": "SUI/USDT", "side": "sell", "score": 0.61}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (stale_run / "decision_audit.json").write_text(
        json.dumps({"run_id": "20260313_13", "regime": "SIDEWAYS"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (stale_run / "strategy_signals.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy": "TrendFollowing",
                        "allocation": 0.2,
                        "total_signals": 2,
                        "buy_signals": 2,
                        "sell_signals": 0,
                        "signals": [{"symbol": "BTC/USDT", "side": "buy", "score": 0.9}],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.utime(stale_run, (1, 1))
    os.utime(current_run, (2, 2))
    monkeypatch.setattr(module, "REPORTS_DIR", reports_dir)

    response = client.get("/api/decision_audit")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["strategy_signal_source"] == "decision_audit"
    assert payload["strategy_run_id"] == "20260313_14"
    assert payload["strategy_signals"][0]["strategy"] == "MeanReversion"


def test_api_decision_audit_falls_back_to_previous_run_strategy_signals(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    reports_dir = tmp_path / "reports"
    runs_dir = reports_dir / "runs"
    current_run = runs_dir / "20260313_15"
    stale_run = runs_dir / "20260313_14"
    current_run.mkdir(parents=True, exist_ok=True)
    stale_run.mkdir(parents=True, exist_ok=True)

    (current_run / "decision_audit.json").write_text(
        json.dumps(
            {"run_id": "20260313_15", "regime": "TRENDING", "counts": {"selected": 0, "orders_rebalance": 0, "orders_exit": 0}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (stale_run / "decision_audit.json").write_text(
        json.dumps({"run_id": "20260313_14", "regime": "SIDEWAYS"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (stale_run / "strategy_signals.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy": "TrendFollowing",
                        "allocation": 0.2,
                        "total_signals": 1,
                        "buy_signals": 1,
                        "sell_signals": 0,
                        "signals": [{"symbol": "BTC/USDT", "side": "buy", "score": 0.9}],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.utime(stale_run, (1, 1))
    os.utime(current_run, (2, 2))
    monkeypatch.setattr(module, "REPORTS_DIR", reports_dir)

    response = client.get("/api/decision_audit")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["strategy_signal_source"] == "previous_run_strategy_file"
    assert payload["strategy_run_id"] == "20260313_14"
    assert payload["strategy_signals"][0]["strategy"] == "TrendFollowing"
    assert payload["fused_source_is_fallback"] is True


def test_shadow_ml_overlay_api_reads_shadow_workspace(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    shadow_workspace = tmp_path / "v5-shadow-tuned-xgboost"
    reports_dir = shadow_workspace / "reports"
    run_dir = reports_dir / "runs" / "shadow_tuned_xgboost_20260313_05"
    runtime_dir = reports_dir / "shadow_tuned_xgboost"
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "decision_audit.json").write_text(
        json.dumps(
            {
                "run_id": "shadow_tuned_xgboost_20260313_05",
                "ml_signal_overview": {
                    "configured_enabled": True,
                    "live_active": True,
                    "prediction_count": 22,
                    "impact_status": "positive",
                    "top_contributors": [
                        {
                            "symbol": "OKB/USDT",
                            "ml_zscore": 2.61,
                            "ml_overlay_score": 1.48,
                            "score_delta": 0.29,
                            "base_rank": 16,
                            "final_rank": 10,
                            "rank_delta": 6,
                        }
                    ],
                    "top_promoted": [
                        {
                            "symbol": "HYPE/USDT",
                            "score_delta": 0.20,
                            "base_rank": 9,
                            "final_rank": 2,
                            "rank_delta": 7,
                        }
                    ],
                    "top_suppressed": [
                        {
                            "symbol": "PIXEL/USDT",
                            "score_delta": -0.25,
                            "base_rank": 3,
                            "final_rank": 4,
                            "rank_delta": -1,
                        }
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (runtime_dir / "ml_runtime_status.json").write_text(
        json.dumps(
            {
                "configured_enabled": True,
                "used_in_latest_snapshot": True,
                "prediction_count": 22,
                "ml_weight": 0.2,
                "overlay_score_max_abs": 1.4836,
                "ts": "2026-03-13T05:15:18Z",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (runtime_dir / "ml_overlay_impact.json").write_text(
        json.dumps(
            {
                "last_step": {"top_n": 3, "delta_bps": 12.4, "status": "positive"},
                "rolling_24h": {"points": 8, "topn_delta_mean_bps": 9.7, "status": "positive"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "_resolve_shadow_workspace", lambda: shadow_workspace)

    response = client.get("/api/shadow_ml_overlay")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["available"] is True
    assert payload["workspace"] == str(shadow_workspace)
    assert payload["run_id"] == "shadow_tuned_xgboost_20260313_05"
    assert payload["ml_signal_overview"]["live_active"] is True
    assert payload["ml_signal_overview"]["overlay_score_max_abs"] == 1.4836
    assert payload["ml_signal_overview"]["rolling_24h"]["topn_delta_mean_bps"] == 9.7
    assert payload["ml_signal_overview"]["top_contributors"][0]["symbol"] == "OKB/USDT"


def test_ml_training_api_reports_four_stage_chain(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    monkeypatch.setattr(module, "WORKSPACE", tmp_path)
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path / "reports")

    reports_dir = module.REPORTS_DIR
    models_dir = module.WORKSPACE / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    db_path = reports_dir / "ml_training_data.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE feature_snapshots(id INTEGER PRIMARY KEY, label_filled INTEGER NOT NULL)")
    cur.executemany(
        "INSERT INTO feature_snapshots(label_filled) VALUES (?)",
        [(1,), (1,), (0,), (1,)],
    )
    conn.commit()
    conn.close()

    (reports_dir / "ml_training_history.json").write_text(
        """
        [
          {
            "timestamp": "2026-03-10T00:30:00Z",
            "valid_ic": 0.12,
            "gate": {"passed": true}
          }
        ]
        """.strip(),
        encoding="utf-8",
    )
    (reports_dir / "model_promotion_decision.json").write_text(
        """
        {
          "ts": "2026-03-10T00:40:00Z",
          "passed": true,
          "fail_reasons": []
        }
        """.strip(),
        encoding="utf-8",
    )
    (reports_dir / "ml_runtime_status.json").write_text(
        """
        {
          "ts": "2026-03-10T01:00:00Z",
          "used_in_latest_snapshot": true,
          "reason": "ok",
          "prediction_count": 3
        }
        """.strip(),
        encoding="utf-8",
    )
    (models_dir / "ml_factor_model.pkl").write_bytes(b"test")
    (models_dir / "ml_factor_model_config.json").write_text("{}", encoding="utf-8")
    (models_dir / "ml_factor_model_active.txt").write_text("models/ml_factor_model", encoding="utf-8")

    response = client.get("/api/ml_training")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["phase"] == "live_active"
    assert payload["status"] == "live_active"
    assert payload["display_status"] == "采样中 是 / 已训练 是 / 已通过门控 是 / 已被实盘使用 是"
    assert payload["stages"] == {
        "sampling": True,
        "trained": True,
        "promoted": True,
        "liveActive": True,
    }
    assert payload["latest_model"] in {"ml_factor_model.pkl", "ml_factor_model_config.json"}
    assert payload["runtime_prediction_count"] == 3


def test_account_api_sanitizes_corrupted_low_peak(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(module, "WORKSPACE", tmp_path)
    monkeypatch.setattr(module, "get_db_connection", lambda: None)
    monkeypatch.setattr(module, "load_config", lambda: {})
    monkeypatch.setattr(module, "api_positions", lambda: module.jsonify({"positions": [{
        "symbol": "OKB",
        "value_usdt": 115.6686,
    }]}))

    reconcile_path = tmp_path / "reconcile_status.json"
    reconcile_path.write_text(
        """
        {
          "exchange_snapshot": {
            "ccy_cashBal": {
              "USDT": 11.48
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    db_path = tmp_path / "positions.sqlite"
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE account_state (
          k TEXT PRIMARY KEY,
          cash_usdt REAL NOT NULL,
          equity_peak_usdt REAL NOT NULL,
          scale_basis_usdt REAL DEFAULT 0.0
        )
        """
    )
    cur.execute(
        "INSERT INTO account_state(k, cash_usdt, equity_peak_usdt, scale_basis_usdt) VALUES ('default', 11.48, 11.48, 0.0)"
    )
    con.commit()
    con.close()

    response = client.get("/api/account")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["cash_usdt"] == 11.48
    assert payload["total_equity_usdt"] == 127.1486
    assert payload["peak_equity_usdt"] == 127.15
    assert payload["drawdown_pct"] == 0.0


def test_account_api_ignores_ambient_live_creds_by_default(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(module, "WORKSPACE", tmp_path)
    monkeypatch.setattr(module, "get_db_connection", lambda: None)
    monkeypatch.setattr(module, "load_config", lambda: {})
    monkeypatch.setattr(module, "api_positions", lambda: module.jsonify({"positions": []}))
    monkeypatch.delenv("V5_DASHBOARD_ALLOW_LIVE_OKX_ACCOUNT", raising=False)
    monkeypatch.delenv("V5_DASHBOARD_ALLOW_LIVE_OKX", raising=False)
    monkeypatch.setenv("EXCHANGE_API_KEY", "ambient-key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "ambient-secret")
    monkeypatch.setenv("EXCHANGE_PASSPHRASE", "ambient-passphrase")

    def _unexpected_request(*args, **kwargs):
        raise AssertionError("api/account should not call live OKX without explicit enable flag")

    monkeypatch.setattr("requests.get", _unexpected_request)

    (tmp_path / "reconcile_status.json").write_text(
        """
        {
          "exchange_snapshot": {
            "ccy_cashBal": {
              "USDT": 11.48
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    response = client.get("/api/account")

    assert response.status_code == 200
    assert response.get_json()["cash_usdt"] == 11.48


def test_trades_api_ignores_ambient_live_creds_by_default(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(module, "WORKSPACE", tmp_path)
    monkeypatch.setattr(module, "get_db_connection", lambda: None)
    monkeypatch.delenv("V5_DASHBOARD_ALLOW_LIVE_OKX_ACCOUNT", raising=False)
    monkeypatch.delenv("V5_DASHBOARD_ALLOW_LIVE_OKX", raising=False)
    monkeypatch.setenv("EXCHANGE_API_KEY", "ambient-key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "ambient-secret")
    monkeypatch.setenv("EXCHANGE_PASSPHRASE", "ambient-passphrase")

    def _unexpected_request(*args, **kwargs):
        raise AssertionError("api/trades should not call live OKX without explicit enable flag")

    monkeypatch.setattr("requests.get", _unexpected_request)

    response = client.get("/api/trades")

    assert response.status_code == 200
    assert response.get_json()["trades"] == []


def test_positions_api_ignores_ambient_live_creds_by_default(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(module, "WORKSPACE", tmp_path)
    monkeypatch.delenv("V5_DASHBOARD_ALLOW_LIVE_OKX_ACCOUNT", raising=False)
    monkeypatch.delenv("V5_DASHBOARD_ALLOW_LIVE_OKX", raising=False)
    monkeypatch.setenv("EXCHANGE_API_KEY", "ambient-key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "ambient-secret")
    monkeypatch.setenv("EXCHANGE_PASSPHRASE", "ambient-passphrase")

    def _unexpected_request(*args, **kwargs):
        raise AssertionError("api/positions should not call live OKX without explicit enable flag")

    monkeypatch.setattr("requests.get", _unexpected_request)

    response = client.get("/api/positions")

    assert response.status_code == 200
    assert response.get_json()["positions"] == []


def test_health_api_ignores_ambient_live_creds_by_default(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(module, "WORKSPACE", tmp_path)
    monkeypatch.setattr(module, "SYSTEMCTL_BIN", None)
    monkeypatch.delenv("V5_DASHBOARD_ALLOW_LIVE_OKX_ACCOUNT", raising=False)
    monkeypatch.delenv("V5_DASHBOARD_ALLOW_LIVE_OKX", raising=False)
    monkeypatch.setenv("EXCHANGE_API_KEY", "ambient-key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "ambient-secret")
    monkeypatch.setenv("EXCHANGE_PASSPHRASE", "ambient-passphrase")

    def _unexpected_request(*args, **kwargs):
        raise AssertionError("api/health should not call live OKX without explicit enable flag")

    monkeypatch.setattr("requests.get", _unexpected_request)

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert any(check.get("name") == "OKX API" and check.get("status") == "warning" for check in payload["checks"])


def test_auto_risk_guard_api_uses_auto_risk_eval_file(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path)

    (tmp_path / "auto_risk_eval.json").write_text(
        """
        {
          "ts": "2026-03-09T22:30:02",
          "current_level": "DEFENSE",
          "metrics": {
            "dd_pct": 0.0682,
            "conversion_rate": 0.0
          },
          "reason": "样本不足 (0轮)，维持当前档位"
        }
        """.strip(),
        encoding="utf-8",
    )

    response = client.get("/api/auto_risk_guard")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["current_level"] == "DEFENSE"
    assert payload["metrics"]["dd_pct"] == 0.0682
    assert payload["reason"] == "样本不足 (0轮)，维持当前档位"
    assert payload["config"]["max_positions"] == 3


def test_market_state_backfills_hmm_vote_from_regime_history(monkeypatch):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    monkeypatch.setattr(module, "_load_market_state_snapshot", lambda _: {
        "state": "TRENDING",
        "position_multiplier": 1.2,
        "method": "decision_audit",
        "votes": {"hmm": {"state": "TRENDING", "confidence": 0.0}},
        "alerts": [],
        "monitor": {},
    })
    monkeypatch.setattr(module, "_load_latest_regime_history_snapshot", lambda _: {
        "votes": {
            "hmm": {
                "state": "TRENDING",
                "confidence": 0.62,
                "weight": 0.35,
                "probs": {"TrendingUp": 0.62, "Sideways": 0.28, "TrendingDown": 0.10},
            }
        }
    })
    monkeypatch.setattr(module, "load_config", lambda: {"regime": {"hmm_weight": 0.35, "funding_weight": 0.4, "rss_weight": 0.25}})
    monkeypatch.setattr(module, "_signal_health", lambda *args, **kwargs: {"status": "fresh", "is_fresh": True, "error": None})
    monkeypatch.setattr(module, "_build_live_funding_vote", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_build_live_rss_vote", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "calculate_market_indicators", lambda: {"price": 0.0})

    response = client.get("/api/market_state")

    assert response.status_code == 200
    payload = response.get_json()
    hmm = payload["votes"]["hmm"]
    assert hmm["confidence"] == 0.62
    assert hmm["weight"] == 0.35
    assert hmm["probs"]["TrendingUp"] == 0.62


def test_market_state_returns_vote_history_and_live_rss_summary(monkeypatch):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    monkeypatch.setattr(module, "_load_market_state_snapshot", lambda _: {
        "state": "SIDEWAYS",
        "position_multiplier": 0.8,
        "method": "decision_audit",
        "votes": {
            "hmm": {"state": "SIDEWAYS", "confidence": 0.7, "probs": {"TrendingUp": 0.1, "Sideways": 0.7, "TrendingDown": 0.2}},
            "rss": {"state": "SIDEWAYS", "confidence": 0.1},
        },
        "alerts": [],
        "monitor": {},
    })
    monkeypatch.setattr(module, "_load_latest_regime_history_snapshot", lambda _: {"votes": {}})
    monkeypatch.setattr(module, "_load_market_vote_history", lambda *args, **kwargs: [{
        "label": "03-09 14:00",
        "final": {"state": "SIDEWAYS", "confidence": 0.3, "score": 0.2},
        "votes": {
            "hmm": {"state": "SIDEWAYS", "confidence": 0.7},
            "funding": {"state": "SIDEWAYS", "confidence": 0.2, "sentiment": 0.01},
            "rss": {"state": "RISK_OFF", "confidence": 0.35, "sentiment": -0.3},
        },
    }])
    monkeypatch.setattr(module, "load_config", lambda: {"regime": {"hmm_weight": 0.35, "funding_weight": 0.4, "rss_weight": 0.25}})
    monkeypatch.setattr(module, "_signal_health", lambda *args, **kwargs: {"status": "fresh", "is_fresh": True, "error": None})
    monkeypatch.setattr(module, "_build_live_funding_vote", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_build_live_rss_vote", lambda *args, **kwargs: {
        "state": "RISK_OFF",
        "confidence": 0.35,
        "weight": 0.25,
        "sentiment": -0.3,
        "summary_short": "\u65b0\u95fb\u504f\u7a7a\uff0c\u4f46\u672a\u5230\u6781\u7aef",
    })
    monkeypatch.setattr(module, "calculate_market_indicators", lambda: {"price": 0.0})

    response = client.get("/api/market_state")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["history_24h"][0]["votes"]["rss"]["state"] == "RISK_OFF"
    assert payload["history_24h"][0]["votes"]["rss"]["sentiment"] == -0.3
    assert payload["votes"]["rss"]["summary_short"] == "\u65b0\u95fb\u504f\u7a7a\uff0c\u4f46\u672a\u5230\u6781\u7aef"


def test_market_state_prefers_live_funding_vote_over_snapshot(monkeypatch):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    monkeypatch.setattr(module, "_load_market_state_snapshot", lambda _: {
        "state": "SIDEWAYS",
        "position_multiplier": 0.8,
        "method": "decision_audit",
        "votes": {
            "hmm": {"state": "SIDEWAYS", "confidence": 0.7, "probs": {"TrendingUp": 0.1, "Sideways": 0.7, "TrendingDown": 0.2}},
            "funding": {"state": "SIDEWAYS", "confidence": 0.54, "sentiment": -0.027},
        },
        "alerts": [],
        "monitor": {},
    })
    monkeypatch.setattr(module, "_load_latest_regime_history_snapshot", lambda _: {"votes": {}})
    monkeypatch.setattr(module, "_load_market_vote_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "load_config", lambda: {"regime": {"hmm_weight": 0.35, "funding_weight": 0.4, "rss_weight": 0.25}})
    monkeypatch.setattr(module, "_signal_health", lambda *args, **kwargs: {"status": "fresh", "is_fresh": True, "error": None})
    monkeypatch.setattr(module, "_build_live_funding_vote", lambda *args, **kwargs: {
        "state": "TRENDING",
        "confidence": 0.18,
        "weight": 0.4,
        "sentiment": 0.09,
        "composite": True,
        "details": {"large": {"avg": 0.1, "count": 2}},
        "raw_state": "TRENDING",
    })
    monkeypatch.setattr(module, "_build_live_rss_vote", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "calculate_market_indicators", lambda: {"price": 0.0})

    response = client.get("/api/market_state")

    assert response.status_code == 200
    payload = response.get_json()
    funding = payload["votes"]["funding"]
    assert funding["state"] == "TRENDING"
    assert funding["confidence"] == 0.18
    assert funding["sentiment"] == 0.09
    assert funding["composite"] is True


def test_market_state_snapshot_falls_back_to_regime_json_after_failed_run(tmp_path):
    module = load_web_dashboard_module()

    reports_dir = tmp_path / "reports"
    runs_dir = reports_dir / "runs"
    failed_run = runs_dir / "20260312_00"
    failed_run.mkdir(parents=True, exist_ok=True)
    (failed_run / "decision_audit.json").write_text(
        json.dumps(
            {
                "regime": "Unknown",
                "top_scores": [],
                "counts": {"universe": 0, "scored": 0},
                "notes": ["No market data returned from provider"],
                "regime_details": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (reports_dir / "regime.json").write_text(
        json.dumps(
            {
                "state": "Trending",
                "multiplier": 1.2,
                "final_score": 0.42,
                "votes": {"hmm": {"state": "TRENDING", "confidence": 0.81}},
                "alerts": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.utime(failed_run, (2, 2))

    snapshot = module._load_market_state_snapshot(reports_dir)

    assert snapshot["state"] == "Trending"
    assert snapshot["method"] == "regime_json"
    assert snapshot["final_score"] == 0.42
    assert snapshot["votes"]["hmm"]["state"] == "TRENDING"


def test_decision_chain_legacy_utc_run_time_is_not_double_shifted(monkeypatch, tmp_path):
    module = load_web_dashboard_module()
    client = module.app.test_client()

    reports_dir = tmp_path / "reports"
    runs_dir = reports_dir / "runs"
    legacy_run = runs_dir / "20260311_00"
    legacy_run.mkdir(parents=True, exist_ok=True)
    (legacy_run / "decision_audit.json").write_text(
        json.dumps(
            {
                "run_id": "20260311_00",
                "now_ts": 3600,
                "regime": "SIDEWAYS",
                "top_scores": [{"symbol": "BTC/USDT", "score": 0.5, "rank": 1}],
                "counts": {"selected": 1, "targets_pre_risk": 1, "orders_rebalance": 0, "orders_exit": 0},
                "router_decisions": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.utime(legacy_run, (0, 0))
    monkeypatch.setattr(module, "REPORTS_DIR", reports_dir)

    response = client.get("/api/decision_chain")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["rounds"][0]["run_id"] == "20260311_00"
    assert payload["rounds"][0]["time"] == "1970-01-01 09:00:00"
