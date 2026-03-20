import json
from datetime import datetime, timezone

import src.core.pipeline as pipeline_module
from configs.schema import AppConfig
from configs.schema import RegimeState
from src.alpha.alpha_engine import AlphaSnapshot
from src.core.clock import FixedClock
from src.core.models import MarketSeries, Order
from src.core.pipeline import V5Pipeline
from src.execution.position_store import Position
from src.portfolio.portfolio_engine import PortfolioSnapshot
from src.regime.regime_engine import RegimeResult
from src.reporting.decision_audit import DecisionAudit


def test_pipeline_marking_and_dd_mult():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pipe = V5Pipeline(AppConfig(symbols=["BTC/USDT"]), clock=FixedClock(t0))

    md = {
        "BTC/USDT": MarketSeries(
            symbol="BTC/USDT",
            timeframe="1h",
            ts=[0],
            open=[100.0],
            high=[110.0],
            low=[90.0],
            close=[105.0],
            volume=[1.0],
        )
    }

    pos = [
        Position(
            symbol="BTC/USDT",
            qty=1.0,
            avg_px=100.0,
            entry_ts=t0.isoformat().replace("+00:00", "Z"),
            highest_px=100.0,
            last_update_ts=t0.isoformat().replace("+00:00", "Z"),
            last_mark_px=100.0,
            unrealized_pnl_pct=0.0,
        )
    ]

    out = pipe.run(md, positions=pos, cash_usdt=1000.0, equity_peak_usdt=1200.0)
    # equity=1000+1*105=1105; peak=1200 => dd~7.9%, no delever => internal scaling should keep dd_mult=1
    dd_mults = [o.meta.get("dd_mult") for o in out.orders if isinstance(o.meta, dict) and "dd_mult" in o.meta]
    assert not dd_mults or all(float(x) == 1.0 for x in dd_mults)


def test_pipeline_ml_snapshot_timestamp_prefers_window_end():
    t0 = datetime(2026, 1, 1, 12, 34, tzinfo=timezone.utc)
    pipe = V5Pipeline(AppConfig(symbols=["BTC/USDT"]), clock=FixedClock(t0))
    audit = DecisionAudit(run_id="20260101_12", window_start_ts=1_704_110_400, window_end_ts=1_704_114_000)

    snapshot_ts = pipe._resolve_ml_snapshot_timestamp_ms(audit=audit)

    assert snapshot_ts == 1_704_114_000_000


def test_pipeline_ml_collection_uses_full_market_universe():
    t0 = datetime(2026, 1, 1, 12, 34, tzinfo=timezone.utc)
    cfg = AppConfig(symbols=["BTC/USDT", "ETH/USDT"])
    pipe = V5Pipeline(cfg, clock=FixedClock(t0))
    md = {
        "BTC/USDT": MarketSeries(
            symbol="BTC/USDT",
            timeframe="1h",
            ts=[0, 1],
            open=[100.0, 100.2],
            high=[101.0, 101.2],
            low=[99.0, 99.2],
            close=[100.5, 100.8],
            volume=[1.0, 1.1],
        ),
        "ETH/USDT": MarketSeries(
            symbol="ETH/USDT",
            timeframe="1h",
            ts=[0, 1],
            open=[200.0, 200.4],
            high=[202.0, 202.4],
            low=[198.0, 198.3],
            close=[201.0, 201.5],
            volume=[1.0, 1.2],
        ),
    }
    audit = DecisionAudit(run_id="20260101_12", window_start_ts=1_704_110_400, window_end_ts=1_704_114_000)
    collected = []

    def fake_collect_features(*, timestamp, symbol, market_data, regime):
        collected.append((timestamp, symbol, regime, len(market_data["close"])))
        return True

    pipe.data_collector.collect_features = fake_collect_features
    pipe.data_collector.fill_labels = lambda current_timestamp: 0
    pipe.portfolio_engine.allocate = lambda **kwargs: PortfolioSnapshot(
        target_weights={"BTC/USDT": 1.0},
        selected=["BTC/USDT"],
        volatilities={},
        entry_candidates=["BTC/USDT"],
    )

    pipe.run(
        md,
        positions=[],
        cash_usdt=1000.0,
        equity_peak_usdt=1000.0,
        audit=audit,
        precomputed_alpha=AlphaSnapshot(raw_factors={}, z_factors={}, scores={"BTC/USDT": 1.0, "ETH/USDT": 0.5}),
        precomputed_regime=RegimeResult(
            state=RegimeState.SIDEWAYS,
            atr_pct=0.0,
            ma20=0.0,
            ma60=0.0,
            multiplier=1.0,
        ),
    )

    assert {(ts, sym) for ts, sym, _, _ in collected} == {
        (1_704_114_000_000, "BTC/USDT"),
        (1_704_114_000_000, "ETH/USDT"),
    }


def test_pipeline_ml_collection_uses_stable_research_universe(tmp_path, monkeypatch):
    t0 = datetime(2026, 1, 1, 12, 34, tzinfo=timezone.utc)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "universe_cache.json").write_text(
        json.dumps({"symbols": ["BTC/USDT", "ETH/USDT"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_module, "REPORTS_DIR", reports_dir)

    cfg = AppConfig(symbols=["BTC/USDT"])
    cfg.execution.ml_research_use_stable_universe = True
    cfg.execution.ml_research_universe_path = "reports/universe_cache.json"

    pipe = V5Pipeline(cfg, clock=FixedClock(t0))
    md = {
        "BTC/USDT": MarketSeries(
            symbol="BTC/USDT",
            timeframe="1h",
            ts=[0, 1],
            open=[100.0, 100.2],
            high=[101.0, 101.2],
            low=[99.0, 99.5],
            close=[100.5, 100.8],
            volume=[1.0, 1.1],
        ),
    }
    audit = DecisionAudit(run_id="20260101_12", window_start_ts=1_704_110_400, window_end_ts=1_704_114_000)
    collected = []

    def fake_collect_features(*, timestamp, symbol, market_data, regime):
        collected.append((timestamp, symbol, regime, len(market_data["close"])))
        return True

    pipe.data_collector.collect_features = fake_collect_features
    pipe.data_collector.fill_labels = lambda current_timestamp: 0
    pipe.data_collector.load_market_data_for_feature_snapshot = lambda symbol, end_timestamp, lookback_bars: {
        "symbol": symbol,
        "ts": [end_timestamp - 3600_000, end_timestamp],
        "open": [200.0, 201.0],
        "high": [202.0, 203.0],
        "low": [198.0, 199.0],
        "close": [201.0, 202.0],
        "volume": [2.0, 2.1],
    } if symbol == "ETH/USDT" else None
    pipe.portfolio_engine.allocate = lambda **kwargs: PortfolioSnapshot(
        target_weights={"BTC/USDT": 1.0},
        selected=["BTC/USDT"],
        volatilities={},
        entry_candidates=["BTC/USDT"],
    )

    pipe.run(
        md,
        positions=[],
        cash_usdt=1000.0,
        equity_peak_usdt=1000.0,
        audit=audit,
        precomputed_alpha=AlphaSnapshot(raw_factors={}, z_factors={}, scores={"BTC/USDT": 1.0, "ETH/USDT": 0.5}),
        precomputed_regime=RegimeResult(
            state=RegimeState.SIDEWAYS,
            atr_pct=0.0,
            ma20=0.0,
            ma60=0.0,
            multiplier=1.0,
        ),
    )

    assert {(ts, sym) for ts, sym, _, _ in collected} == {
        (1_704_114_000_000, "BTC/USDT"),
        (1_704_114_000_000, "ETH/USDT"),
    }


def test_rebalance_turnover_cap_uses_side_turnover_budget():
    cfg = AppConfig(symbols=["BTC/USDT"])
    cfg.execution.max_rebalance_turnover_per_cycle = 0.30
    pipe = V5Pipeline(cfg, clock=FixedClock(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    orders = [
        Order(
            symbol="OLD/USDT",
            side="sell",
            intent="REBALANCE",
            notional_usdt=20.0,
            signal_price=1.0,
            meta={"drift": -0.20},
        ),
        Order(
            symbol="ADD/USDT",
            side="buy",
            intent="REBALANCE",
            notional_usdt=15.0,
            signal_price=1.0,
            meta={"drift": 0.30},
        ),
        Order(
            symbol="NEW/USDT",
            side="buy",
            intent="OPEN_LONG",
            notional_usdt=15.0,
            signal_price=1.0,
            meta={"drift": 0.35},
        ),
    ]

    kept, dropped, stats = pipe._apply_rebalance_turnover_cap(orders, equity_raw=100.0)

    assert not dropped
    assert [order.symbol for order in kept] == ["OLD/USDT", "ADD/USDT", "NEW/USDT"]
    assert stats["effective_turnover_notional"] == 30.0
    assert stats["cap_notional"] == 30.0


def test_rebalance_turnover_cap_prioritizes_new_opens_before_existing_buys():
    cfg = AppConfig(symbols=["BTC/USDT"])
    cfg.execution.max_rebalance_turnover_per_cycle = 0.25
    pipe = V5Pipeline(cfg, clock=FixedClock(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    orders = [
        Order(
            symbol="OPEN_BIG/USDT",
            side="buy",
            intent="OPEN_LONG",
            notional_usdt=20.0,
            signal_price=1.0,
            meta={"drift": 0.60},
        ),
        Order(
            symbol="ADD_WINNER/USDT",
            side="buy",
            intent="REBALANCE",
            notional_usdt=15.0,
            signal_price=1.0,
            meta={"drift": 0.30},
        ),
        Order(
            symbol="OPEN_SMALL/USDT",
            side="buy",
            intent="OPEN_LONG",
            notional_usdt=5.0,
            signal_price=1.0,
            meta={"drift": 0.20},
        ),
    ]

    kept, dropped, stats = pipe._apply_rebalance_turnover_cap(orders, equity_raw=100.0)

    assert [order.symbol for order in kept] == ["OPEN_BIG/USDT", "OPEN_SMALL/USDT"]
    assert [order.symbol for order in dropped] == ["ADD_WINNER/USDT"]
    assert stats["kept_buy_notional"] == 25.0


def test_rebalance_turnover_cap_clips_oversized_open_to_cap_for_top_ranked_entry():
    cfg = AppConfig(symbols=["BTC/USDT"])
    cfg.execution.max_rebalance_turnover_per_cycle = 0.25
    pipe = V5Pipeline(cfg, clock=FixedClock(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    orders = [
        Order(
            symbol="OPEN_OVERSIZED/USDT",
            side="buy",
            intent="OPEN_LONG",
            notional_usdt=30.0,
            signal_price=1.0,
            meta={"drift": 0.80, "score_rank": 1, "clip_min_notional": 25.0},
        ),
        Order(
            symbol="OPEN_FIT/USDT",
            side="buy",
            intent="OPEN_LONG",
            notional_usdt=20.0,
            signal_price=1.0,
            meta={"drift": 0.50, "score_rank": 2, "clip_min_notional": 25.0},
        ),
        Order(
            symbol="ADD_SMALL/USDT",
            side="buy",
            intent="REBALANCE",
            notional_usdt=5.0,
            signal_price=1.0,
            meta={"drift": 0.20, "score_rank": 3, "clip_min_notional": 25.0},
        ),
    ]

    kept, dropped, stats = pipe._apply_rebalance_turnover_cap(orders, equity_raw=100.0)

    assert [order.symbol for order in kept] == ["OPEN_OVERSIZED/USDT"]
    assert [order.symbol for order in dropped] == ["OPEN_FIT/USDT", "ADD_SMALL/USDT"]
    assert stats["cap_notional"] == 25.0
    assert stats["kept_buy_notional"] == 25.0
    assert stats["clipped_buy_count"] == 1.0
    assert kept[0].notional_usdt == 25.0
    assert kept[0].meta["turnover_cap_clipped"] is True
    assert kept[0].meta["turnover_cap_original_notional"] == 30.0
    assert kept[0].meta["turnover_cap_clipped_notional"] == 25.0


def test_rebalance_turnover_cap_drops_oversized_open_when_remaining_below_clip_floor():
    cfg = AppConfig(symbols=["BTC/USDT"])
    cfg.execution.max_rebalance_turnover_per_cycle = 0.20
    pipe = V5Pipeline(cfg, clock=FixedClock(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    orders = [
        Order(
            symbol="OPEN_OVERSIZED/USDT",
            side="buy",
            intent="OPEN_LONG",
            notional_usdt=30.0,
            signal_price=1.0,
            meta={"drift": 0.80, "score_rank": 1, "clip_min_notional": 25.0},
        ),
        Order(
            symbol="OPEN_FIT/USDT",
            side="buy",
            intent="OPEN_LONG",
            notional_usdt=20.0,
            signal_price=1.0,
            meta={"drift": 0.50, "score_rank": 2, "clip_min_notional": 25.0},
        ),
    ]

    kept, dropped, stats = pipe._apply_rebalance_turnover_cap(orders, equity_raw=100.0)

    assert [order.symbol for order in kept] == ["OPEN_FIT/USDT"]
    assert [order.symbol for order in dropped] == ["OPEN_OVERSIZED/USDT"]
    assert stats["cap_notional"] == 20.0
    assert stats["kept_buy_notional"] == 20.0
    assert stats["clipped_buy_count"] == 0.0


def test_rebalance_turnover_cap_keeps_higher_ranked_buy_before_lower_ranked_buy():
    cfg = AppConfig(symbols=["BTC/USDT"])
    cfg.execution.max_rebalance_turnover_per_cycle = 0.25
    pipe = V5Pipeline(cfg, clock=FixedClock(datetime(2026, 1, 1, tzinfo=timezone.utc)))
    orders = [
        Order(
            symbol="TOP1/USDT",
            side="buy",
            intent="OPEN_LONG",
            notional_usdt=20.0,
            signal_price=1.0,
            meta={"drift": 0.20, "score_rank": 1},
        ),
        Order(
            symbol="TOP4/USDT",
            side="buy",
            intent="OPEN_LONG",
            notional_usdt=20.0,
            signal_price=1.0,
            meta={"drift": 0.60, "score_rank": 4},
        ),
        Order(
            symbol="TOP6/USDT",
            side="buy",
            intent="REBALANCE",
            notional_usdt=5.0,
            signal_price=1.0,
            meta={"drift": 0.80, "score_rank": 6},
        ),
    ]

    kept, dropped, stats = pipe._apply_rebalance_turnover_cap(orders, equity_raw=100.0)

    assert [order.symbol for order in kept] == ["TOP1/USDT", "TOP6/USDT"]
    assert [order.symbol for order in dropped] == ["TOP4/USDT"]
    assert stats["kept_buy_notional"] == 25.0


def test_pipeline_prefers_in_memory_strategy_signal_payload():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cfg = AppConfig(symbols=["BTC/USDT"])
    pipe = V5Pipeline(cfg, clock=FixedClock(t0))
    pipe.alpha_engine.use_multi_strategy = True
    pipe.alpha_engine.get_latest_strategy_signal_payload = lambda: {
        "run_id": "20260101_00",
        "strategies": [
            {
                "strategy": "MeanReversion",
                "total_signals": 1,
                "buy_signals": 0,
                "sell_signals": 1,
                "allocation": 0.25,
                "signals": [{"symbol": "BTC/USDT", "side": "sell", "score": 0.66}],
            }
        ],
    }
    pipe.alpha_engine.strategy_signals_path = lambda: None
    pipe.portfolio_engine.allocate = lambda **kwargs: PortfolioSnapshot(
        target_weights={"BTC/USDT": 1.0},
        selected=["BTC/USDT"],
        volatilities={},
        entry_candidates=["BTC/USDT"],
    )
    pipe.data_collector.collect_features = lambda **kwargs: True
    pipe.data_collector.fill_labels = lambda current_timestamp: 0

    md = {
        "BTC/USDT": MarketSeries(
            symbol="BTC/USDT",
            timeframe="1h",
            ts=[0, 1],
            open=[100.0, 101.0],
            high=[101.0, 102.0],
            low=[99.0, 100.0],
            close=[100.5, 101.5],
            volume=[1.0, 1.0],
        )
    }
    audit = DecisionAudit(run_id="20260101_00")

    pipe.run(
        md,
        positions=[],
        cash_usdt=1000.0,
        equity_peak_usdt=1000.0,
        audit=audit,
        precomputed_alpha=AlphaSnapshot(raw_factors={}, z_factors={}, scores={"BTC/USDT": 1.0}),
        precomputed_regime=RegimeResult(
            state=RegimeState.SIDEWAYS,
            atr_pct=0.0,
            ma20=0.0,
            ma60=0.0,
            multiplier=1.0,
        ),
    )

    assert audit.strategy_signals[0]["strategy"] == "MeanReversion"
    assert "source=memory" in " ".join(audit.notes)


def test_pipeline_uses_audit_run_id_when_run_logger_missing():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cfg = AppConfig(symbols=["BTC/USDT"])
    pipe = V5Pipeline(cfg, clock=FixedClock(t0))
    captured = {}

    pipe.alpha_engine.set_run_id = lambda run_id: captured.__setitem__("alpha", run_id)
    pipe.portfolio_engine.set_run_id = lambda run_id: captured.__setitem__("portfolio", run_id)
    pipe.portfolio_engine.allocate = lambda **kwargs: PortfolioSnapshot(
        target_weights={"BTC/USDT": 1.0},
        selected=["BTC/USDT"],
        volatilities={},
        entry_candidates=["BTC/USDT"],
    )
    pipe.data_collector.collect_features = lambda **kwargs: True
    pipe.data_collector.fill_labels = lambda current_timestamp: 0

    md = {
        "BTC/USDT": MarketSeries(
            symbol="BTC/USDT",
            timeframe="1h",
            ts=[0, 1],
            open=[100.0, 101.0],
            high=[101.0, 102.0],
            low=[99.0, 100.0],
            close=[100.5, 101.5],
            volume=[1.0, 1.0],
        )
    }
    audit = DecisionAudit(run_id="20260101_00")

    pipe.run(
        md,
        positions=[],
        cash_usdt=1000.0,
        equity_peak_usdt=1000.0,
        audit=audit,
        precomputed_alpha=AlphaSnapshot(raw_factors={}, z_factors={}, scores={"BTC/USDT": 1.0}),
        precomputed_regime=RegimeResult(
            state=RegimeState.SIDEWAYS,
            atr_pct=0.0,
            ma20=0.0,
            ma60=0.0,
            multiplier=1.0,
        ),
    )

    assert captured["alpha"] == "20260101_00"
    assert captured["portfolio"] == "20260101_00"


def test_pipeline_records_ml_rank_delta_and_impact_summary(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pipeline_module, "REPORTS_DIR", reports_dir)

    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    pipe = V5Pipeline(AppConfig(symbols=["AAA/USDT", "BBB/USDT", "CCC/USDT"]), clock=FixedClock(t0))
    pipe.portfolio_engine.allocate = lambda **kwargs: PortfolioSnapshot(
        target_weights={"AAA/USDT": 0.5, "BBB/USDT": 0.5},
        selected=["AAA/USDT", "BBB/USDT"],
        volatilities={},
        entry_candidates=["AAA/USDT", "BBB/USDT"],
    )
    pipe.data_collector.collect_features = lambda **kwargs: True
    pipe.data_collector.fill_labels = lambda current_timestamp: 0

    md1 = {
        "AAA/USDT": MarketSeries(symbol="AAA/USDT", timeframe="1h", ts=[0], open=[10.0], high=[10.0], low=[10.0], close=[10.0], volume=[1.0]),
        "BBB/USDT": MarketSeries(symbol="BBB/USDT", timeframe="1h", ts=[0], open=[20.0], high=[20.0], low=[20.0], close=[20.0], volume=[1.0]),
        "CCC/USDT": MarketSeries(symbol="CCC/USDT", timeframe="1h", ts=[0], open=[30.0], high=[30.0], low=[30.0], close=[30.0], volume=[1.0]),
    }
    alpha1 = AlphaSnapshot(
        raw_factors={},
        z_factors={},
        scores={"BBB/USDT": 0.9, "AAA/USDT": 0.7, "CCC/USDT": 0.1},
        raw_scores={"BBB/USDT": 0.9, "AAA/USDT": 0.7, "CCC/USDT": 0.1},
        telemetry_scores={"AAA/USDT": 0.6, "BBB/USDT": 0.4, "CCC/USDT": 0.2},
        base_scores={"AAA/USDT": 0.8, "BBB/USDT": 0.6, "CCC/USDT": 0.1},
        base_raw_scores={"AAA/USDT": 0.8, "BBB/USDT": 0.6, "CCC/USDT": 0.1},
        ml_attribution_scores={"BBB/USDT": 0.9, "AAA/USDT": 0.7, "CCC/USDT": 0.1},
        ml_overlay_scores={"AAA/USDT": 0.3, "BBB/USDT": 1.8},
        ml_overlay_raw_scores={"AAA/USDT": 0.5, "BBB/USDT": 2.7},
        ml_runtime={
            "configured_enabled": True,
            "promotion_passed": True,
            "used_in_latest_snapshot": True,
            "prediction_count": 2,
            "ml_weight": 0.2,
            "configured_ml_weight": 0.2,
            "effective_ml_weight": 0.2,
            "overlay_mode": "live",
            "reason": "ok",
            "ts": "2026-01-01T12:00:00Z",
            "overlay_transform": "tanh",
            "overlay_transform_scale": 1.6,
            "overlay_transform_max_abs": 1.6,
            "overlay_score_max_abs": 1.6,
        },
    )
    audit1 = DecisionAudit(run_id="20260101_12", window_start_ts=1_704_110_400, window_end_ts=1_704_114_000)

    pipe.run(
        md1,
        positions=[],
        cash_usdt=100.0,
        equity_peak_usdt=100.0,
        audit=audit1,
        precomputed_alpha=alpha1,
        precomputed_regime=RegimeResult(state=RegimeState.SIDEWAYS, atr_pct=0.0, ma20=0.0, ma60=0.0, multiplier=1.0),
    )

    assert audit1.top_scores[0]["symbol"] == "BBB/USDT"
    assert audit1.top_scores[0]["base_rank"] == 2
    assert audit1.top_scores[0]["rank_delta"] == 1
    assert audit1.ml_signal_overview["top_promoted"][0]["symbol"] == "BBB/USDT"

    pipe.clock = FixedClock(datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc))
    md2 = {
        "AAA/USDT": MarketSeries(symbol="AAA/USDT", timeframe="1h", ts=[0], open=[10.2], high=[10.2], low=[10.2], close=[10.2], volume=[1.0]),
        "BBB/USDT": MarketSeries(symbol="BBB/USDT", timeframe="1h", ts=[0], open=[21.0], high=[21.0], low=[21.0], close=[21.0], volume=[1.0]),
        "CCC/USDT": MarketSeries(symbol="CCC/USDT", timeframe="1h", ts=[0], open=[29.7], high=[29.7], low=[29.7], close=[29.7], volume=[1.0]),
    }
    alpha2 = AlphaSnapshot(
        raw_factors={},
        z_factors={},
        scores={"BBB/USDT": 0.8, "AAA/USDT": 0.6, "CCC/USDT": 0.2},
        raw_scores={"BBB/USDT": 0.8, "AAA/USDT": 0.6, "CCC/USDT": 0.2},
        telemetry_scores={"AAA/USDT": 0.5, "BBB/USDT": 0.3, "CCC/USDT": 0.1},
        base_scores={"AAA/USDT": 0.7, "BBB/USDT": 0.55, "CCC/USDT": 0.2},
        base_raw_scores={"AAA/USDT": 0.7, "BBB/USDT": 0.55, "CCC/USDT": 0.2},
        ml_attribution_scores={"BBB/USDT": 0.8, "AAA/USDT": 0.6, "CCC/USDT": 0.2},
        ml_overlay_scores={"AAA/USDT": 0.25, "BBB/USDT": 1.4},
        ml_overlay_raw_scores={"AAA/USDT": 0.4, "BBB/USDT": 2.1},
        ml_runtime=alpha1.ml_runtime,
    )
    audit2 = DecisionAudit(run_id="20260101_13", window_start_ts=1_704_114_000, window_end_ts=1_704_117_600)

    pipe.run(
        md2,
        positions=[],
        cash_usdt=100.0,
        equity_peak_usdt=100.0,
        audit=audit2,
        precomputed_alpha=alpha2,
        precomputed_regime=RegimeResult(state=RegimeState.SIDEWAYS, atr_pct=0.0, ma20=0.0, ma60=0.0, multiplier=1.0),
    )

    impact_summary = json.loads((reports_dir / "ml_overlay_impact.json").read_text(encoding="utf-8"))
    assert impact_summary["last_step"]["delta_bps"] is not None
    assert impact_summary["rolling_24h"]["points"] >= 1
    assert impact_summary["rolling_48h"]["points"] >= 1
    assert audit2.ml_signal_overview["last_step"]["delta_bps"] == impact_summary["last_step"]["delta_bps"]
    assert audit2.ml_signal_overview["overlay_mode"] == "live"


def test_pipeline_ml_impact_uses_attribution_scores_in_shadow_mode(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pipeline_module, "REPORTS_DIR", reports_dir)

    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    pipe = V5Pipeline(AppConfig(symbols=["AAA/USDT", "BBB/USDT", "CCC/USDT"]), clock=FixedClock(t0))
    pipe.portfolio_engine.allocate = lambda **kwargs: PortfolioSnapshot(
        target_weights={"AAA/USDT": 0.5, "BBB/USDT": 0.5},
        selected=["AAA/USDT", "BBB/USDT"],
        volatilities={},
        entry_candidates=["AAA/USDT", "BBB/USDT"],
    )
    pipe.data_collector.collect_features = lambda **kwargs: True
    pipe.data_collector.fill_labels = lambda current_timestamp: 0

    md = {
        "AAA/USDT": MarketSeries(symbol="AAA/USDT", timeframe="1h", ts=[0], open=[10.0], high=[10.0], low=[10.0], close=[10.0], volume=[1.0]),
        "BBB/USDT": MarketSeries(symbol="BBB/USDT", timeframe="1h", ts=[0], open=[20.0], high=[20.0], low=[20.0], close=[20.0], volume=[1.0]),
        "CCC/USDT": MarketSeries(symbol="CCC/USDT", timeframe="1h", ts=[0], open=[30.0], high=[30.0], low=[30.0], close=[30.0], volume=[1.0]),
    }
    alpha = AlphaSnapshot(
        raw_factors={},
        z_factors={},
        scores={"AAA/USDT": 0.8, "BBB/USDT": 0.6, "CCC/USDT": 0.1},
        raw_scores={"AAA/USDT": 0.8, "BBB/USDT": 0.6, "CCC/USDT": 0.1},
        telemetry_scores={"AAA/USDT": 0.8, "BBB/USDT": 0.6, "CCC/USDT": 0.1},
        base_scores={"AAA/USDT": 0.8, "BBB/USDT": 0.6, "CCC/USDT": 0.1},
        base_raw_scores={"AAA/USDT": 0.8, "BBB/USDT": 0.6, "CCC/USDT": 0.1},
        ml_attribution_scores={"BBB/USDT": 0.95, "AAA/USDT": 0.7, "CCC/USDT": 0.1},
        ml_overlay_scores={"AAA/USDT": 0.1, "BBB/USDT": 1.2},
        ml_overlay_raw_scores={"AAA/USDT": 0.2, "BBB/USDT": 2.0},
        ml_runtime={
            "configured_enabled": True,
            "promotion_passed": True,
            "used_in_latest_snapshot": False,
            "prediction_count": 2,
            "ml_weight": 0.2,
            "configured_ml_weight": 0.2,
            "effective_ml_weight": 0.0,
            "overlay_mode": "shadow",
            "reason": "ok",
            "ts": "2026-01-01T12:00:00Z",
        },
    )
    audit = DecisionAudit(run_id="20260101_12", window_start_ts=1_704_110_400, window_end_ts=1_704_114_000)

    pipe.run(
        md,
        positions=[],
        cash_usdt=100.0,
        equity_peak_usdt=100.0,
        audit=audit,
        precomputed_alpha=alpha,
        precomputed_regime=RegimeResult(state=RegimeState.SIDEWAYS, atr_pct=0.0, ma20=0.0, ma60=0.0, multiplier=1.0),
    )

    assert audit.top_scores[0]["symbol"] == "AAA/USDT"
    assert audit.ml_signal_overview["overlay_mode"] == "shadow"
    assert audit.ml_signal_overview["top_promoted"][0]["symbol"] == "BBB/USDT"


def test_pipeline_buy_sizing_respects_target_gap_when_target_gross_is_below_one():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cfg = AppConfig(symbols=["AAA/USDT", "BBB/USDT"])
    pipe = V5Pipeline(cfg, clock=FixedClock(t0))

    md = {
        "AAA/USDT": MarketSeries(
            symbol="AAA/USDT",
            timeframe="1h",
            ts=[0, 1],
            open=[1.0, 1.0],
            high=[1.0, 1.0],
            low=[1.0, 1.0],
            close=[1.0, 1.0],
            volume=[1.0, 1.0],
        ),
        "BBB/USDT": MarketSeries(
            symbol="BBB/USDT",
            timeframe="1h",
            ts=[0, 1],
            open=[1.0, 1.0],
            high=[1.0, 1.0],
            low=[1.0, 1.0],
            close=[1.0, 1.0],
            volume=[1.0, 1.0],
        ),
    }

    pipe.portfolio_engine.allocate = lambda **kwargs: PortfolioSnapshot(
        target_weights={"AAA/USDT": 0.25, "BBB/USDT": 0.25},
        selected=["AAA/USDT", "BBB/USDT"],
        volatilities={},
        entry_candidates=["AAA/USDT", "BBB/USDT"],
    )

    out = pipe.run(
        md,
        positions=[],
        cash_usdt=100.0,
        equity_peak_usdt=100.0,
        precomputed_alpha=AlphaSnapshot(
            raw_factors={},
            z_factors={},
            scores={"AAA/USDT": 1.0, "BBB/USDT": 0.9},
        ),
        precomputed_regime=RegimeResult(
            state=RegimeState.SIDEWAYS,
            atr_pct=0.0,
            ma20=0.0,
            ma60=0.0,
            multiplier=1.0,
        ),
    )

    buys = sorted((order.symbol, float(order.notional_usdt)) for order in out.orders if order.side == "buy")

    assert buys == [("AAA/USDT", 25.0), ("BBB/USDT", 25.0)]
    assert sum(notional for _, notional in buys) == 50.0
