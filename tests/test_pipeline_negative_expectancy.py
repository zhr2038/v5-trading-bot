from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.schema import AppConfig, RegimeState
from src.alpha.alpha_engine import AlphaSnapshot
from src.core.models import MarketSeries
from src.core.pipeline import V5Pipeline
from src.execution.position_store import Position
from src.regime.regime_engine import RegimeResult
from src.reporting.decision_audit import DecisionAudit


def _ms(ts_s: int) -> int:
    return ts_s * 1000


def _series(sym: str, close: float) -> MarketSeries:
    ts = [_ms(1700000000 + i * 3600) for i in range(30)]
    close_arr = [close for _ in range(30)]
    return MarketSeries(
        symbol=sym,
        timeframe="1h",
        ts=ts,
        open=close_arr,
        high=close_arr,
        low=close_arr,
        close=close_arr,
        volume=[1000.0 for _ in range(30)],
    )


def _regime() -> RegimeResult:
    return RegimeResult(
        state=RegimeState.SIDEWAYS,
        atr_pct=0.01,
        ma20=100.0,
        ma60=95.0,
        multiplier=1.0,
    )


def _build_pipe(cfg: AppConfig) -> V5Pipeline:
    pipe = V5Pipeline(cfg)
    pipe.exit_policy.evaluate = lambda **kwargs: []
    pipe.stop_loss_manager.register_position = lambda *args, **kwargs: None
    pipe.stop_loss_manager.evaluate_stop = lambda *args, **kwargs: (False, 0.0, "", 0.0)
    pipe.fixed_stop_loss.register_position = lambda *args, **kwargs: None
    pipe.fixed_stop_loss.should_stop_loss = lambda *args, **kwargs: (False, 0.0, 0.0)
    pipe.data_collector.collect_features = lambda **kwargs: None
    pipe.data_collector.fill_labels = lambda current_ts: 0
    return pipe


def test_pipeline_applies_negative_expectancy_penalty_before_allocate():
    cfg = AppConfig(symbols=["OKB/USDT", "HYPE/USDT"])
    cfg.alpha.use_fused_score_for_weighting = False
    cfg.execution.negative_expectancy_score_penalty_enabled = True
    cfg.execution.negative_expectancy_score_penalty_min_closed_cycles = 2
    cfg.execution.negative_expectancy_score_penalty_floor_bps = 5.0
    cfg.execution.negative_expectancy_score_penalty_per_bps = 0.015
    cfg.execution.negative_expectancy_score_penalty_max = 0.60
    cfg.execution.negative_expectancy_open_block_enabled = False
    cfg.execution.negative_expectancy_cooldown_enabled = False

    pipe = _build_pipe(cfg)
    captured = {}

    def _allocate(scores, market_data, regime_mult, audit=None):
        captured.update(scores)
        return SimpleNamespace(
            target_weights={},
            selected=[],
            entry_candidates=[],
            volatilities={},
            notes="",
        )

    pipe.portfolio_engine.allocate = _allocate
    pipe.negative_expectancy_cooldown.refresh = lambda force=False: {
        "stats": {
            "OKB/USDT": {
                "closed_cycles": 3,
                "expectancy_bps": -10.0,
                "pnl_sum_usdt": -0.30,
                "closed_notional_usdt": 100.0,
            }
        },
        "symbols": {},
    }

    market_data = {
        "OKB/USDT": _series("OKB/USDT", 100.0),
        "HYPE/USDT": _series("HYPE/USDT", 30.0),
    }
    alpha = AlphaSnapshot(raw_factors={}, z_factors={}, scores={"OKB/USDT": 0.60, "HYPE/USDT": 0.40})

    pipe.run(
        market_data_1h=market_data,
        positions=[],
        cash_usdt=100.0,
        equity_peak_usdt=100.0,
        audit=DecisionAudit(run_id="neg-penalty"),
        precomputed_alpha=alpha,
        precomputed_regime=_regime(),
    )

    assert captured["OKB/USDT"] == pytest.approx(0.375)
    assert captured["HYPE/USDT"] == pytest.approx(0.40)


def test_pipeline_respects_zero_negative_expectancy_penalty_cap():
    cfg = AppConfig(symbols=["OKB/USDT", "HYPE/USDT"])
    cfg.alpha.use_fused_score_for_weighting = False
    cfg.execution.negative_expectancy_score_penalty_enabled = True
    cfg.execution.negative_expectancy_score_penalty_min_closed_cycles = 2
    cfg.execution.negative_expectancy_score_penalty_floor_bps = 5.0
    cfg.execution.negative_expectancy_score_penalty_per_bps = 0.015
    cfg.execution.negative_expectancy_score_penalty_max = 0.0
    cfg.execution.negative_expectancy_open_block_enabled = False
    cfg.execution.negative_expectancy_cooldown_enabled = False

    pipe = _build_pipe(cfg)
    captured = {}

    def _allocate(scores, market_data, regime_mult, audit=None):
        captured.update(scores)
        return SimpleNamespace(
            target_weights={},
            selected=[],
            entry_candidates=[],
            volatilities={},
            notes="",
        )

    pipe.portfolio_engine.allocate = _allocate
    pipe.negative_expectancy_cooldown.refresh = lambda force=False: {
        "stats": {
            "OKB/USDT": {
                "closed_cycles": 3,
                "expectancy_bps": -10.0,
                "pnl_sum_usdt": -0.30,
                "closed_notional_usdt": 100.0,
            }
        },
        "symbols": {},
    }

    market_data = {
        "OKB/USDT": _series("OKB/USDT", 100.0),
        "HYPE/USDT": _series("HYPE/USDT", 30.0),
    }
    alpha = AlphaSnapshot(raw_factors={}, z_factors={}, scores={"OKB/USDT": 0.60, "HYPE/USDT": 0.40})

    pipe.run(
        market_data_1h=market_data,
        positions=[],
        cash_usdt=100.0,
        equity_peak_usdt=100.0,
        audit=DecisionAudit(run_id="neg-penalty-zero-cap"),
        precomputed_alpha=alpha,
        precomputed_regime=_regime(),
    )

    assert captured["OKB/USDT"] == pytest.approx(0.60)
    assert captured["HYPE/USDT"] == pytest.approx(0.40)


def test_pipeline_blocks_open_long_on_negative_expectancy():
    cfg = AppConfig(symbols=["OKB/USDT"])
    cfg.alpha.use_fused_score_for_weighting = False
    cfg.execution.negative_expectancy_score_penalty_enabled = False
    cfg.execution.negative_expectancy_open_block_enabled = True
    cfg.execution.negative_expectancy_open_block_min_closed_cycles = 2
    cfg.execution.negative_expectancy_open_block_floor_bps = 5.0
    cfg.execution.negative_expectancy_cooldown_enabled = False

    pipe = _build_pipe(cfg)
    pipe.portfolio_engine.allocate = lambda scores, market_data, regime_mult, audit=None: SimpleNamespace(
        target_weights={"OKB/USDT": 0.25},
        selected=["OKB/USDT"],
        entry_candidates=["OKB/USDT"],
        volatilities={},
        notes="",
    )
    pipe.negative_expectancy_cooldown.refresh = lambda force=False: {
        "stats": {
            "OKB/USDT": {
                "closed_cycles": 3,
                "expectancy_bps": 1.0,
                "pnl_sum_usdt": 0.01,
                "closed_notional_usdt": 100.0,
            }
        },
        "symbols": {},
    }
    pipe.negative_expectancy_cooldown.get_symbol_stats = lambda symbol: {
        "closed_cycles": 3,
        "expectancy_bps": 1.0,
        "cooldown_active": False,
    }

    market_data = {"OKB/USDT": _series("OKB/USDT", 100.0)}
    alpha = AlphaSnapshot(raw_factors={}, z_factors={}, scores={"OKB/USDT": 0.60})
    audit = DecisionAudit(run_id="neg-open-block")

    out = pipe.run(
        market_data_1h=market_data,
        positions=[],
        cash_usdt=100.0,
        equity_peak_usdt=100.0,
        audit=audit,
        precomputed_alpha=alpha,
        precomputed_regime=_regime(),
    )

    assert out.orders == []
    assert any(
        d.get("symbol") == "OKB/USDT" and d.get("reason") == "negative_expectancy_open_block"
        for d in audit.router_decisions
    )


def test_pipeline_respects_zero_negative_expectancy_open_block_floor():
    cfg = AppConfig(symbols=["OKB/USDT"])
    cfg.alpha.use_fused_score_for_weighting = False
    cfg.execution.negative_expectancy_score_penalty_enabled = False
    cfg.execution.negative_expectancy_open_block_enabled = True
    cfg.execution.negative_expectancy_open_block_min_closed_cycles = 2
    cfg.execution.negative_expectancy_open_block_floor_bps = 0.0
    cfg.execution.negative_expectancy_cooldown_enabled = False

    pipe = _build_pipe(cfg)
    pipe.portfolio_engine.allocate = lambda scores, market_data, regime_mult, audit=None: SimpleNamespace(
        target_weights={"OKB/USDT": 0.25},
        selected=["OKB/USDT"],
        entry_candidates=["OKB/USDT"],
        volatilities={},
        notes="",
    )
    pipe.negative_expectancy_cooldown.refresh = lambda force=False: {
        "stats": {
            "OKB/USDT": {
                "closed_cycles": 3,
                "expectancy_bps": 1.0,
                "pnl_sum_usdt": 0.01,
                "closed_notional_usdt": 100.0,
            }
        },
        "symbols": {},
    }
    pipe.negative_expectancy_cooldown.get_symbol_stats = lambda symbol: {
        "closed_cycles": 3,
        "expectancy_bps": 1.0,
        "cooldown_active": False,
    }

    market_data = {"OKB/USDT": _series("OKB/USDT", 100.0)}
    alpha = AlphaSnapshot(raw_factors={}, z_factors={}, scores={"OKB/USDT": 0.60})
    audit = DecisionAudit(run_id="neg-open-floor-zero")

    out = pipe.run(
        market_data_1h=market_data,
        positions=[],
        cash_usdt=100.0,
        equity_peak_usdt=100.0,
        audit=audit,
        precomputed_alpha=alpha,
        precomputed_regime=_regime(),
    )

    assert any(o.symbol == "OKB/USDT" and o.side == "buy" for o in out.orders)
    assert not any(
        d.get("symbol") == "OKB/USDT" and d.get("reason") == "negative_expectancy_open_block"
        for d in audit.router_decisions
    )


def test_pipeline_blocks_open_long_on_negative_expectancy_fast_fail():
    cfg = AppConfig(symbols=["ROBO/USDT"])
    cfg.alpha.use_fused_score_for_weighting = False
    cfg.execution.negative_expectancy_score_penalty_enabled = False
    cfg.execution.negative_expectancy_open_block_enabled = False
    cfg.execution.negative_expectancy_fast_fail_open_block_enabled = True
    cfg.execution.negative_expectancy_fast_fail_open_block_min_closed_cycles = 2
    cfg.execution.negative_expectancy_fast_fail_open_block_floor_bps = 0.0
    cfg.execution.negative_expectancy_fast_fail_max_hold_minutes = 120
    cfg.execution.negative_expectancy_cooldown_enabled = False

    pipe = _build_pipe(cfg)
    pipe.portfolio_engine.allocate = lambda scores, market_data, regime_mult, audit=None: SimpleNamespace(
        target_weights={"ROBO/USDT": 0.25},
        selected=["ROBO/USDT"],
        entry_candidates=["ROBO/USDT"],
        volatilities={},
        notes="",
    )
    pipe.negative_expectancy_cooldown.refresh = lambda force=False: {
        "stats": {
            "ROBO/USDT": {
                "closed_cycles": 3,
                "expectancy_bps": 20.0,
                "fast_fail_closed_cycles": 2,
                "fast_fail_expectancy_bps": -120.0,
                "fast_fail_avg_hold_minutes": 64.0,
            }
        },
        "symbols": {},
    }
    pipe.negative_expectancy_cooldown.get_symbol_stats = lambda symbol: {
        "closed_cycles": 3,
        "expectancy_bps": 20.0,
        "fast_fail_closed_cycles": 2,
        "fast_fail_expectancy_bps": -120.0,
        "fast_fail_avg_hold_minutes": 64.0,
        "cooldown_active": False,
    }

    market_data = {"ROBO/USDT": _series("ROBO/USDT", 0.08)}
    alpha = AlphaSnapshot(raw_factors={}, z_factors={}, scores={"ROBO/USDT": 0.60})
    audit = DecisionAudit(run_id="neg-fast-fail-block")

    out = pipe.run(
        market_data_1h=market_data,
        positions=[],
        cash_usdt=100.0,
        equity_peak_usdt=100.0,
        audit=audit,
        precomputed_alpha=alpha,
        precomputed_regime=_regime(),
    )

    assert out.orders == []
    assert any(
        d.get("symbol") == "ROBO/USDT" and d.get("reason") == "negative_expectancy_fast_fail_open_block"
        for d in audit.router_decisions
    )


def test_pipeline_respects_zero_anti_chase_max_premium():
    cfg = AppConfig(symbols=["OKB/USDT"])
    cfg.alpha.use_fused_score_for_weighting = False
    cfg.execution.negative_expectancy_score_penalty_enabled = False
    cfg.execution.negative_expectancy_open_block_enabled = False
    cfg.execution.negative_expectancy_fast_fail_open_block_enabled = False
    cfg.execution.negative_expectancy_cooldown_enabled = False
    cfg.execution.anti_chase_enabled = True
    cfg.execution.anti_chase_max_entry_premium_pct = 0.0
    cfg.execution.anti_chase_max_add_notional_ratio = 1.0

    pipe = _build_pipe(cfg)
    pipe.portfolio_engine.allocate = lambda scores, market_data, regime_mult, audit=None: SimpleNamespace(
        target_weights={"OKB/USDT": 1.0},
        selected=["OKB/USDT"],
        entry_candidates=["OKB/USDT"],
        volatilities={},
        notes="",
    )
    pipe.negative_expectancy_cooldown.refresh = lambda force=False: {"stats": {}, "symbols": {}}
    pipe.negative_expectancy_cooldown.get_symbol_stats = lambda symbol: {}

    positions = [
        Position(
            symbol="OKB/USDT",
            qty=1.0,
            avg_px=99.5,
            entry_ts="2026-03-20T00:00:00Z",
            highest_px=100.0,
            last_update_ts="2026-03-20T00:00:00Z",
            last_mark_px=100.0,
            unrealized_pnl_pct=0.0,
        )
    ]
    market_data = {"OKB/USDT": _series("OKB/USDT", 100.0)}
    alpha = AlphaSnapshot(raw_factors={}, z_factors={}, scores={"OKB/USDT": 0.60})
    audit = DecisionAudit(run_id="anti-chase-premium-zero")

    out = pipe.run(
        market_data_1h=market_data,
        positions=positions,
        cash_usdt=20.0,
        equity_peak_usdt=120.0,
        audit=audit,
        precomputed_alpha=alpha,
        precomputed_regime=_regime(),
    )

    assert out.orders == []
    assert any(
        d.get("symbol") == "OKB/USDT" and d.get("reason") == "anti_chase_premium"
        for d in audit.router_decisions
    )


def test_pipeline_respects_zero_anti_chase_add_ratio():
    cfg = AppConfig(symbols=["OKB/USDT"])
    cfg.alpha.use_fused_score_for_weighting = False
    cfg.execution.negative_expectancy_score_penalty_enabled = False
    cfg.execution.negative_expectancy_open_block_enabled = False
    cfg.execution.negative_expectancy_fast_fail_open_block_enabled = False
    cfg.execution.negative_expectancy_cooldown_enabled = False
    cfg.execution.anti_chase_enabled = True
    cfg.execution.anti_chase_max_entry_premium_pct = 1.0
    cfg.execution.anti_chase_max_add_notional_ratio = 0.0

    pipe = _build_pipe(cfg)
    pipe.portfolio_engine.allocate = lambda scores, market_data, regime_mult, audit=None: SimpleNamespace(
        target_weights={"OKB/USDT": 1.0},
        selected=["OKB/USDT"],
        entry_candidates=["OKB/USDT"],
        volatilities={},
        notes="",
    )
    pipe.negative_expectancy_cooldown.refresh = lambda force=False: {"stats": {}, "symbols": {}}
    pipe.negative_expectancy_cooldown.get_symbol_stats = lambda symbol: {}

    positions = [
        Position(
            symbol="OKB/USDT",
            qty=1.0,
            avg_px=100.0,
            entry_ts="2026-03-20T00:00:00Z",
            highest_px=100.0,
            last_update_ts="2026-03-20T00:00:00Z",
            last_mark_px=100.0,
            unrealized_pnl_pct=0.0,
        )
    ]
    market_data = {"OKB/USDT": _series("OKB/USDT", 100.0)}
    alpha = AlphaSnapshot(raw_factors={}, z_factors={}, scores={"OKB/USDT": 0.60})
    audit = DecisionAudit(run_id="anti-chase-add-zero")

    out = pipe.run(
        market_data_1h=market_data,
        positions=positions,
        cash_usdt=20.0,
        equity_peak_usdt=120.0,
        audit=audit,
        precomputed_alpha=alpha,
        precomputed_regime=_regime(),
    )

    assert out.orders == []
    assert any(
        d.get("symbol") == "OKB/USDT" and d.get("reason") == "anti_chase_add_size"
        for d in audit.router_decisions
    )


def test_pipeline_demotes_negative_expectancy_blocked_symbol_before_ranking():
    cfg = AppConfig(symbols=["ROBO/USDT", "WLD/USDT"])
    cfg.alpha.use_fused_score_for_weighting = False
    cfg.execution.negative_expectancy_score_penalty_enabled = False
    cfg.execution.negative_expectancy_open_block_enabled = False
    cfg.execution.negative_expectancy_fast_fail_open_block_enabled = True
    cfg.execution.negative_expectancy_fast_fail_open_block_min_closed_cycles = 2
    cfg.execution.negative_expectancy_fast_fail_open_block_floor_bps = 0.0
    cfg.execution.negative_expectancy_fast_fail_max_hold_minutes = 120
    cfg.execution.negative_expectancy_cooldown_enabled = False

    pipe = _build_pipe(cfg)
    captured = {}

    def _allocate(scores, market_data, regime_mult, audit=None):
        captured.update(scores)
        return SimpleNamespace(
            target_weights={},
            selected=[],
            entry_candidates=[],
            volatilities={},
            notes="",
        )

    pipe.portfolio_engine.allocate = _allocate
    pipe.negative_expectancy_cooldown.refresh = lambda force=False: {
        "stats": {
            "ROBO/USDT": {
                "closed_cycles": 3,
                "expectancy_bps": 20.0,
                "fast_fail_closed_cycles": 2,
                "fast_fail_expectancy_bps": -120.0,
                "fast_fail_avg_hold_minutes": 64.0,
            }
        },
        "symbols": {},
    }

    market_data = {
        "ROBO/USDT": _series("ROBO/USDT", 0.08),
        "WLD/USDT": _series("WLD/USDT", 1.8),
    }
    alpha = AlphaSnapshot(raw_factors={}, z_factors={}, scores={"ROBO/USDT": 0.90, "WLD/USDT": 0.50})
    audit = DecisionAudit(run_id="neg-rank-guard")

    pipe.run(
        market_data_1h=market_data,
        positions=[],
        cash_usdt=100.0,
        equity_peak_usdt=100.0,
        audit=audit,
        precomputed_alpha=alpha,
        precomputed_regime=_regime(),
    )

    assert captured["ROBO/USDT"] < captured["WLD/USDT"]
    assert audit.top_scores[0]["symbol"] == "WLD/USDT"
    assert any("NegativeExpectancy rank-guard: ROBO/USDT" in note for note in audit.notes)


def test_pipeline_low_price_entry_guard_raises_cost_floor():
    cfg = AppConfig(symbols=["PUMP/USDT"])
    cfg.alpha.use_fused_score_for_weighting = False
    cfg.execution.cost_aware_entry_enabled = True
    cfg.execution.cost_aware_score_per_bps = 0.003
    cfg.execution.cost_aware_min_score_floor = 0.14
    cfg.execution.cost_aware_roundtrip_cost_bps = 22.0
    cfg.execution.low_price_entry_guard_enabled = True
    cfg.execution.low_price_entry_threshold_usdt = 0.05
    cfg.execution.low_price_entry_extra_score_floor = 0.08
    cfg.execution.low_price_entry_extra_cost_bps = 12.0
    cfg.execution.negative_expectancy_score_penalty_enabled = False
    cfg.execution.negative_expectancy_open_block_enabled = False
    cfg.execution.negative_expectancy_cooldown_enabled = False

    pipe = _build_pipe(cfg)
    pipe.portfolio_engine.allocate = lambda scores, market_data, regime_mult, audit=None: SimpleNamespace(
        target_weights={"PUMP/USDT": 0.25},
        selected=["PUMP/USDT"],
        entry_candidates=["PUMP/USDT"],
        volatilities={},
        notes="",
    )

    market_data = {"PUMP/USDT": _series("PUMP/USDT", 0.0021)}
    alpha = AlphaSnapshot(raw_factors={}, z_factors={}, scores={"PUMP/USDT": 0.26})
    audit = DecisionAudit(run_id="low-price-guard")

    out = pipe.run(
        market_data_1h=market_data,
        positions=[],
        cash_usdt=100.0,
        equity_peak_usdt=100.0,
        audit=audit,
        precomputed_alpha=alpha,
        precomputed_regime=_regime(),
    )

    assert out.orders == []
    assert any(
        d.get("symbol") == "PUMP/USDT"
        and d.get("reason") == "cost_aware_edge"
        and d.get("low_price_guard_applied") is True
        for d in audit.router_decisions
    )


def test_pipeline_normal_price_entry_not_hit_by_low_price_guard():
    cfg = AppConfig(symbols=["BTC/USDT"])
    cfg.alpha.use_fused_score_for_weighting = False
    cfg.execution.cost_aware_entry_enabled = True
    cfg.execution.cost_aware_score_per_bps = 0.003
    cfg.execution.cost_aware_min_score_floor = 0.14
    cfg.execution.cost_aware_roundtrip_cost_bps = 22.0
    cfg.execution.low_price_entry_guard_enabled = True
    cfg.execution.low_price_entry_threshold_usdt = 0.05
    cfg.execution.low_price_entry_extra_score_floor = 0.08
    cfg.execution.low_price_entry_extra_cost_bps = 12.0
    cfg.execution.negative_expectancy_score_penalty_enabled = False
    cfg.execution.negative_expectancy_open_block_enabled = False
    cfg.execution.negative_expectancy_cooldown_enabled = False

    pipe = _build_pipe(cfg)
    pipe.portfolio_engine.allocate = lambda scores, market_data, regime_mult, audit=None: SimpleNamespace(
        target_weights={"BTC/USDT": 0.25},
        selected=["BTC/USDT"],
        entry_candidates=["BTC/USDT"],
        volatilities={},
        notes="",
    )

    market_data = {"BTC/USDT": _series("BTC/USDT", 70000.0)}
    alpha = AlphaSnapshot(raw_factors={}, z_factors={}, scores={"BTC/USDT": 0.26})
    audit = DecisionAudit(run_id="normal-price-guard")

    out = pipe.run(
        market_data_1h=market_data,
        positions=[],
        cash_usdt=100.0,
        equity_peak_usdt=100.0,
        audit=audit,
        precomputed_alpha=alpha,
        precomputed_regime=_regime(),
    )

    assert len(out.orders) == 1
    assert out.orders[0].symbol == "BTC/USDT"
