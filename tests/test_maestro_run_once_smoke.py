from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from maestro.config.loader import load_config


def test_live_approval_example_loads_with_kis_broker_products():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs/fugue_kis_live_approval_dry_run.example.yaml")

    assert [item.value for item in config.kis.effective_broker_products()] == ["kis_overseas_stock"]


def test_maestro_run_once_loads_tradingagents_adapter_and_normalizes_signal(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    symphony_root = repo_root.parents[1]
    maestro_root = symphony_root / "Maestro"
    config_path = tmp_path / "fugue_mock_paper.yaml"
    state_path = tmp_path / "state.db"
    audit_path = tmp_path / "audit.jsonl"

    config_path.write_text(
        textwrap.dedent(
            f"""
            mode: paper

            portfolio:
              base_currency: KRW
              initial_cash: 10000000
              allowed_symbols:
                - CASH
                - MOCK_ETF_A

            strategies:
              - id: fugue
                enabled: true
                weight: 1.0
                entrypoint: "fugue.strategy:FugueStrategy"
                signal_to_allocation:
                  type: single_symbol_action_map
                  cash_symbol: CASH
                  action_target_weights:
                    buy: 0.30
                    hold: 0.10
                    sell: 0.0
                config:
                  symbol: MOCK_ETF_A
                  asset_type: domestic_etf
                  cash_symbol: CASH
                  selected_analysts: ["market", "news", "fundamentals"]

            datahub:
              provider: mock

            execution:
              proposal_engine: paper
              order_generation_mode: target_rebalance

            state:
              sqlite_path: {state_path}

            audit:
              jsonl_path: {audit_path}

            approval:
              enabled: false
              provider: console
              require_approval: false
              default_decision: approved
              timeout_seconds: 300
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    script = textwrap.dedent(
        """
        import json
        import sys
        from pathlib import Path

        from maestro.config.loader import load_config
        from maestro.orchestration.orchestrator import MaestroOrchestrator
        from maestro.sdk import DataRequest
        from maestro.state.store import StateStore
        import fugue.strategy as adapter


        class FakeTradingAgentsGraph:
            def __init__(self, selected_analysts, config):
                assert selected_analysts == ["market", "news", "fundamentals"]
                assert config["data_vendors"]["core_stock_apis"] == "maestro"

            def propagate(self, symbol, trade_date):
                assert symbol == "MOCK_ETF_A"
                assert trade_date
                decision = "Rating: Buy\\nSynthetic Maestro smoke signal."
                return (
                    {
                        "final_trade_decision": decision,
                        "market_report": "deterministic market report",
                        "news_report": "deterministic news report",
                        "fundamentals_report": "deterministic fundamentals report",
                        "investment_plan": "deterministic investment plan",
                    },
                    decision,
                )


        def fake_build_data_requests(self, context):
            del self
            return [
                DataRequest(
                    symbol=context.config["symbol"],
                    asset_type=context.config["asset_type"],
                    data_type="price",
                    intended_use="tradable",
                )
            ]


        adapter.TradingAgentsGraph = FakeTradingAgentsGraph
        adapter.FugueStrategy.build_data_requests = fake_build_data_requests

        config = load_config(Path(sys.argv[1]))
        summary = MaestroOrchestrator(config).run_once()
        store = StateStore(
            config.state.sqlite_path,
            config.portfolio.initial_cash,
            config.portfolio.cash_by_currency,
        )
        strategy_run = store.list_strategy_runs(limit=1)[0]["payload"]
        orders = store.list_orders(limit=10)

        assert summary.loaded_strategies == ["fugue"]
        assert summary.orders_created == 1
        assert strategy_run["source_signal"]["symbol"] == "MOCK_ETF_A"
        assert strategy_run["source_signal"]["action"] == "buy"
        assert strategy_run["source_signal"]["rating"] == "Buy"
        assert strategy_run["result"]["allocations"] == {"MOCK_ETF_A": 0.3, "CASH": 0.7}
        assert strategy_run["result"]["metadata"]["source_signal"]["action"] == "buy"
        assert len(orders) == 1
        assert orders[0]["payload"]["symbol"] == "MOCK_ETF_A"

        print(
            json.dumps(
                {
                    "loaded_strategies": summary.loaded_strategies,
                    "orders_created": summary.orders_created,
                    "allocations": strategy_run["result"]["allocations"],
                    "source_signal": strategy_run["source_signal"],
                },
                sort_keys=True,
            )
        )
        """
    )

    env = os.environ.copy()
    pythonpath = [
        str(maestro_root / "src"),
        str(repo_root),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(path for path in pythonpath if path)

    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["loaded_strategies"] == ["fugue"]
    assert payload["orders_created"] == 1
    assert payload["allocations"] == {"MOCK_ETF_A": 0.3, "CASH": 0.7}
    assert payload["source_signal"]["action"] == "buy"


def test_maestro_live_approval_dry_run_loads_tradingagents_adapter(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    symphony_root = repo_root.parents[1]
    maestro_root = symphony_root / "Maestro"
    config_path = tmp_path / "fugue_mock_live_approval.yaml"
    state_path = tmp_path / "state.db"
    audit_path = tmp_path / "audit.jsonl"

    config_path.write_text(
        textwrap.dedent(
            f"""
            mode: live_approval

            portfolio:
              base_currency: USD
              cash_by_currency:
                USD: 1000
              allowed_symbols:
                - CASH
                - MOCK_ETF_A

            universe:
              instruments:
                - symbol: CASH
                  asset_type: cash
                  region: US
                  currency: USD
                  broker: kis
                  broker_product: kis_overseas_stock
                  broker_symbol: USD
                  quantity_step: 0.01
                  price_tick: 0.01
                  min_order_quantity: 0.01
                  min_order_notional: 0
                - symbol: MOCK_ETF_A
                  asset_type: us_etf
                  region: US
                  currency: USD
                  broker: kis
                  broker_product: kis_overseas_stock
                  broker_symbol: MOCK_ETF_A
                  exchange_code: NASD
                  quantity_step: 1
                  price_tick: 0.01
                  min_order_quantity: 1
                  min_order_notional: 1

            strategies:
              - id: fugue
                enabled: true
                weight: 1.0
                entrypoint: "fugue.strategy:FugueStrategy"
                signal_to_allocation:
                  type: single_symbol_action_map
                  cash_symbol: CASH
                  action_target_weights:
                    buy: 0.30
                    hold: 0.10
                    sell: 0.0
                config:
                  symbol: MOCK_ETF_A
                  asset_type: us_etf
                  cash_symbol: CASH
                  selected_analysts: ["market", "news", "fundamentals"]

            datahub:
              provider: mock

            execution:
              proposal_engine: paper
              order_posture: dry_run
              order_generation_mode: target_rebalance
              require_reconciliation_pass: true
              live_order_limits:
                max_order_notional: 500
                max_daily_notional: 1000
                max_daily_order_count: 1
              market_session:
                required: false
              broker_validation:
                require_quote_validation: false
                require_risk_validation: false

            state:
              sqlite_path: {state_path}

            audit:
              jsonl_path: {audit_path}

            approval:
              enabled: true
              provider: telegram
              require_approval: true
              default_decision: expired
              timeout_seconds: 1
              telegram_allowed_chat_ids: [100]
              whitelisted_user_ids: [100]
              telegram_poll_interval_seconds: 0.0

            kis:
              enabled: true
              provider: mock
              broker_products:
                - kis_overseas_stock
              account_id: MOCK

            reconciliation:
              cash_tolerance: 0.0
              position_quantity_tolerance: 0.0
              value_tolerance: 0.0
              max_age_seconds: 86400
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    script = textwrap.dedent(
        """
        import json
        import sys
        from pathlib import Path

        import maestro.approval.manager as approval_manager
        from maestro.config.loader import load_config
        from maestro.core.clock import utc_now
        from maestro.execution.reconciliation import ReconciliationResult
        from maestro.orchestration.orchestrator import MaestroOrchestrator
        from maestro.plugins.loader import load_strategy
        from maestro.sdk import DataRequest
        from maestro.state.models import PortfolioState
        from maestro.state.store import StateStore
        import fugue.strategy as adapter


        APPROVAL_ID = "appr_fugue_live_dry_run"


        class FakeTelegramClient:
            def __init__(self):
                self.sent_messages = []

            def send_message(self, chat_id, text, reply_markup=None):
                self.sent_messages.append(
                    {"chat_id": chat_id, "text": text, "reply_markup": reply_markup}
                )
                return {"ok": True, "result": {"message_id": len(self.sent_messages)}}

            def get_updates(self, *, offset, timeout_seconds):
                return {
                    "ok": True,
                    "result": [
                        {
                            "update_id": 1,
                            "callback_query": {
                                "id": "callback-1",
                                "data": f"approve:{APPROVAL_ID}",
                                "message": {
                                    "chat": {"id": 100},
                                    "message_id": 1,
                                    "text": "Maestro approval request",
                                },
                                "from": {"id": 100, "username": "approver"},
                            },
                        }
                    ],
                }


        class FakeBrokerReconciliation:
            def reconcile_latest(self):
                return ReconciliationResult(
                    run_id="run_broker_reconcile",
                    passed=True,
                    checked_at=utc_now().isoformat(),
                    issues=[],
                    tolerances={
                        "cash_tolerance": 0.0,
                        "position_quantity_tolerance": 0.0,
                        "value_tolerance": 0.0,
                    },
                )


        class FakeTradingAgentsGraph:
            def __init__(self, selected_analysts, config):
                assert selected_analysts == ["market", "news", "fundamentals"]
                assert config["data_vendors"]["core_stock_apis"] == "maestro"

            def propagate(self, symbol, trade_date):
                assert symbol == "MOCK_ETF_A"
                assert trade_date
                decision = "Rating: Buy\\nSynthetic live approval dry-run signal."
                return (
                    {
                        "final_trade_decision": decision,
                        "market_report": "deterministic market report",
                        "news_report": "deterministic news report",
                        "fundamentals_report": "deterministic fundamentals report",
                        "investment_plan": "deterministic investment plan",
                    },
                    decision,
                )


        def fake_build_data_requests(self, context):
            del self
            return [
                DataRequest(
                    symbol=context.config["symbol"],
                    asset_type=context.config["asset_type"],
                    data_type="price",
                    intended_use="tradable",
                )
            ]


        approval_manager.new_approval_id = lambda: APPROVAL_ID
        adapter.TradingAgentsGraph = FakeTradingAgentsGraph
        adapter.FugueStrategy.build_data_requests = fake_build_data_requests

        config = load_config(Path(sys.argv[1]))
        plugin = load_strategy(config.strategies[0], run_mode=config.mode)
        manifest = plugin.manifest()
        assert config.mode == "live_approval"
        assert manifest.supported_modes == ["paper", "live_approval"]
        assert manifest.can_run_live is True

        orchestrator = MaestroOrchestrator(
            config,
            broker_reconciliation_service=FakeBrokerReconciliation(),
            telegram_client=FakeTelegramClient(),
        )
        orchestrator.state_store.save_portfolio_snapshot(
            "run_adopt_broker_snapshot",
            PortfolioState(cash=1000.0, cash_by_currency={"USD": 1000.0}, positions={}),
        )
        orchestrator.state_store.save_system_event(
            "run_reconcile_initial",
            "broker_reconciliation",
            {"passed": True},
        )
        summary = orchestrator.run_once()

        store = StateStore(
            config.state.sqlite_path,
            config.portfolio.initial_cash,
            config.portfolio.cash_by_currency,
        )
        strategy_run = store.list_strategy_runs(limit=1)[0]["payload"]
        proposal = store.list_system_events_by_type("live_proposal_data_snapshot")[0][
            "payload"
        ]
        dry_run = store.list_system_events_by_type("live_order_dry_run")[0]["payload"]

        assert summary.loaded_strategies == ["fugue"]
        assert summary.orders_created == 1
        assert strategy_run["source_signal"]["symbol"] == "MOCK_ETF_A"
        assert strategy_run["source_signal"]["action"] == "buy"
        assert strategy_run["source_signal"]["rating"] == "Buy"
        assert strategy_run["result"]["allocations"] == {"MOCK_ETF_A": 0.3, "CASH": 0.7}
        assert proposal["order_prices"] == {"MOCK_ETF_A": 100.0}
        assert proposal["proposed_orders"][0]["symbol"] == "MOCK_ETF_A"
        assert dry_run["broker_submit_skipped"] is True
        assert dry_run["request"]["symbol"] == "MOCK_ETF_A"
        assert dry_run["approval_decision"]["status"] == "approved"
        assert store.list_orders() == []

        print(
            json.dumps(
                {
                    "loaded_strategies": summary.loaded_strategies,
                    "orders_created": summary.orders_created,
                    "allocations": strategy_run["result"]["allocations"],
                    "dry_run_symbol": dry_run["request"]["symbol"],
                },
                sort_keys=True,
            )
        )
        """
    )

    env = os.environ.copy()
    pythonpath = [
        str(maestro_root / "src"),
        str(repo_root),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(path for path in pythonpath if path)

    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["loaded_strategies"] == ["fugue"]
    assert payload["orders_created"] == 1
    assert payload["allocations"] == {"MOCK_ETF_A": 0.3, "CASH": 0.7}
    assert payload["dry_run_symbol"] == "MOCK_ETF_A"
