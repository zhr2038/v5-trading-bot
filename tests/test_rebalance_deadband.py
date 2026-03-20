from __future__ import annotations

from types import SimpleNamespace

import pytest

from configs.schema import AppConfig
from src.core.models import MarketSeries
from src.core.pipeline import V5Pipeline
from src.execution.position_store import Position
from src.reporting.decision_audit import DecisionAudit


def _ms(ts_s: int) -> int:
    return ts_s * 1000


def _series(sym: str, close: float) -> MarketSeries:
    # minimal MarketSeries with enough bars for alpha (>=25)
    ts = [_ms(1700000000 + i * 3600) for i in range(30)]
    close_arr = [close for _ in range(30)]
    vol = [1000.0 for _ in range(30)]
    return MarketSeries(symbol=sym, timeframe="1h", ts=ts, open=close_arr, high=close_arr, low=close_arr, close=close_arr, volume=vol)


def test_deadband_skips_small_drift():
    cfg = AppConfig()
    cfg.rebalance.deadband_sideways = 0.05
    cfg.budget.min_trade_notional_base = 0.0
    cfg.budget.exchange_min_notional_enabled = False

    pipe = V5Pipeline(cfg)

    md = {"SOL/USDT": _series("SOL/USDT", 100.0), "BTC/USDT": _series("BTC/USDT", 100.0)}

    # current weight ~0.23, target 0.25 => drift 0.02 <= 0.05 -> skip
    positions = [
        Position(
            symbol="SOL/USDT",
            qty=0.23,
            avg_px=100.0,
            entry_ts="t",
            highest_px=100.0,
            last_update_ts="t",
            last_mark_px=100.0,
            unrealized_pnl_pct=0.0,
        )
    ]
    audit = DecisionAudit(run_id="t")

    # disable exits for this unit test
    pipe.exit_policy.evaluate = lambda **kwargs: []

    # force target by monkeypatching portfolio_engine to return desired target
    pipe.portfolio_engine.allocate = lambda scores, market_data, regime_mult, audit=None: SimpleNamespace(
        target_weights={"SOL/USDT": 0.25}, selected=["SOL/USDT"], volatilities={}, notes=""
    )

    out = pipe.run(market_data_1h=md, positions=positions, cash_usdt=77.0, equity_peak_usdt=100.0, audit=audit)

    assert len(out.orders) == 0
    assert audit.rebalance_deadband_pct == pytest.approx(0.05)
    assert audit.rebalance_skipped_deadband_count == 1
    assert "SOL/USDT" in audit.rebalance_skipped_deadband_by_symbol


def test_deadband_allows_large_drift():
    cfg = AppConfig()
    cfg.rebalance.deadband_sideways = 0.05
    cfg.budget.min_trade_notional_base = 0.0
    cfg.budget.exchange_min_notional_enabled = False

    pipe = V5Pipeline(cfg)

    md = {"SOL/USDT": _series("SOL/USDT", 100.0), "BTC/USDT": _series("BTC/USDT", 100.0)}

    # current weight 0.10, target 0.25 => drift 0.15 > 0.05 -> allow
    positions = [
        Position(
            symbol="SOL/USDT",
            qty=0.10,
            avg_px=100.0,
            entry_ts="t",
            highest_px=100.0,
            last_update_ts="t",
            last_mark_px=100.0,
            unrealized_pnl_pct=0.0,
        )
    ]
    audit = DecisionAudit(run_id="t")

    # disable exits for this unit test
    pipe.exit_policy.evaluate = lambda **kwargs: []

    pipe.portfolio_engine.allocate = lambda scores, market_data, regime_mult, audit=None: SimpleNamespace(
        target_weights={"SOL/USDT": 0.25}, selected=["SOL/USDT"], volatilities={}, notes=""
    )

    out = pipe.run(market_data_1h=md, positions=positions, cash_usdt=90.0, equity_peak_usdt=100.0, audit=audit)

    # should create one rebalance/open order
    assert len(out.orders) == 1
    assert out.orders[0].symbol == "SOL/USDT"
    assert audit.rebalance_skipped_deadband_count == 0
