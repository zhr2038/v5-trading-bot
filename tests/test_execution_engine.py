from configs.schema import ExecutionConfig
from src.core.models import Order
from src.execution.account_store import AccountState, AccountStore
from src.execution.execution_engine import ExecutionEngine
from src.execution.position_store import PositionStore


class _TradeLogRecorder:
    def __init__(self):
        self.fills = []

    def append_fill(self, fill):
        self.fills.append(fill)


def test_dry_run_rebalance_sell_keeps_remaining_position(tmp_path):
    db_path = tmp_path / "positions.sqlite"
    position_store = PositionStore(str(db_path))
    account_store = AccountStore(str(db_path))
    trade_log = _TradeLogRecorder()

    position_store.upsert_buy("BTC/USDT", qty=1.0, px=100.0, now_ts="2026-03-20T00:00:00Z")
    account_store.set(AccountState(cash_usdt=0.0, equity_peak_usdt=100.0, scale_basis_usdt=0.0))

    engine = ExecutionEngine(
        ExecutionConfig(
            slippage_db_path=str(tmp_path / "slippage.sqlite"),
            fee_bps=0.0,
            slippage_bps=0.0,
        ),
        position_store=position_store,
        account_store=account_store,
        trade_log=trade_log,
        run_id="unit-test",
    )

    engine.execute(
        [
            Order(
                symbol="BTC/USDT",
                side="sell",
                intent="REBALANCE",
                notional_usdt=40.0,
                signal_price=100.0,
                meta={},
            )
        ]
    )

    remaining = position_store.get("BTC/USDT")
    account = account_store.get()

    assert remaining is not None
    assert remaining.qty == 0.6
    assert account.cash_usdt == 40.0
    assert len(trade_log.fills) == 1
    assert trade_log.fills[0].qty == 0.4
    assert trade_log.fills[0].notional_usdt == 40.0
