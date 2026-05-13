from __future__ import annotations

import ast
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest


@dataclass
class StrategyManifest:
    sdk_contract_version: str
    strategy_id: str
    name: str
    version: str
    supported_modes: list[str]
    supported_asset_types: list[str]
    result_type: str
    requires_data: list[str]
    requires_llm: bool = False
    supported_llm_providers: list[str] | None = None
    required_env_vars: list[str] | None = None
    can_run_live: bool = False
    allow_direct_external_data_calls: bool = False
    estimated_runtime_seconds: int | None = None


@dataclass
class DataRequest:
    symbol: str
    asset_type: str
    data_type: str
    intended_use: str
    timeframe: str | None = None
    lookback: int | None = None
    start: datetime | None = None
    end: datetime | None = None
    as_of: str | None = None
    indicator: str | None = None
    limit: int | None = None
    query: str | None = None
    statement_type: str | None = None
    frequency: str | None = None


@dataclass
class TargetAllocationResult:
    strategy_id: str
    strategy_version: str
    timestamp: datetime
    allocations: dict[str, float]
    confidence: float
    time_horizon: str
    rationale: str
    metadata: dict


class BaseStrategyPlugin:
    pass


@dataclass
class StrategyContext:
    strategy_id: str = "tradingagents"
    timestamp: datetime = datetime(2025, 1, 15, 12, 0, 0)
    run_mode: str = "paper"
    config: dict | None = None


@dataclass
class DataBundle:
    data: dict


class StrategyRuntime:
    pass


sdk = types.ModuleType("maestro.sdk")
sdk.BaseStrategyPlugin = BaseStrategyPlugin
sdk.DataBundle = DataBundle
sdk.DataRequest = DataRequest
sdk.StrategyContext = StrategyContext
sdk.StrategyManifest = StrategyManifest
sdk.StrategyRuntime = StrategyRuntime
sdk.TargetAllocationResult = TargetAllocationResult

maestro = types.ModuleType("maestro")
maestro.sdk = sdk
sys.modules.setdefault("maestro", maestro)
sys.modules.setdefault("maestro.sdk", sdk)

from tradingagents.dataflows import config as dataflow_config
from tradingagents.dataflows import interface as dataflow_interface
from tradingagents_virtuoso import strategy as adapter
from tradingagents_virtuoso.strategy import TradingAgentsVirtuosoStrategy


def _context(**overrides):
    config = {
        "symbol": "AAPL",
        "asset_type": "stock",
        "cash_symbol": "CASH",
    }
    config.update(overrides)
    return StrategyContext(config=config)


def test_manifest_matches_maestro_contract():
    plugin = TradingAgentsVirtuosoStrategy()

    manifest = plugin.manifest()

    assert isinstance(plugin, BaseStrategyPlugin)
    assert manifest.strategy_id == "tradingagents"
    assert manifest.sdk_contract_version == "1.1"
    assert manifest.result_type == "target_allocation"
    assert manifest.supported_modes == ["paper"]
    assert "financial_statements" in manifest.requires_data
    assert manifest.requires_llm is True
    assert manifest.allow_direct_external_data_calls is False


def test_build_data_requests_declares_prefetch_contract():
    plugin = TradingAgentsVirtuosoStrategy()

    requests = plugin.build_data_requests(
        _context(include_insider_transactions=True, news_limit=5)
    )

    by_key = {(req.symbol, req.data_type): req for req in requests}
    assert by_key[("AAPL", "ohlcv")].intended_use == "tradable"
    assert by_key[("AAPL", "ohlcv")].lookback == 260
    assert by_key[("AAPL", "price")].asset_type == "stock"
    assert by_key[("AAPL", "fundamental")].intended_use == "tradable"
    assert by_key[("AAPL", "news")].limit == 5
    assert by_key[("MARKET", "news")].intended_use == "research"
    assert by_key[("AAPL", "insider_transactions")].intended_use == "tradable"

    statement_requests = [
        req for req in requests if req.symbol == "AAPL" and req.data_type == "financial_statements"
    ]
    assert {req.statement_type for req in statement_requests} == {
        "balance_sheet",
        "cashflow",
        "income_statement",
    }
    assert {req.frequency for req in statement_requests} == {"annual"}


@pytest.mark.parametrize(
    ("decision", "expected_weight", "expected_confidence"),
    [
        ("Rating: Buy\nAdd exposure.", 0.30, 0.75),
        ("Rating: Hold\nWait for clarity.", 0.10, 0.50),
        ("Rating: Sell\nExit risk.", 0.0, 0.75),
    ],
)
def test_run_maps_tradingagents_rating_to_target_allocation(
    monkeypatch, decision, expected_weight, expected_confidence
):
    class FakeGraph:
        def __init__(self, selected_analysts, config):
            self.selected_analysts = selected_analysts
            self.config = config

        def propagate(self, symbol, trade_date):
            assert symbol == "AAPL"
            assert trade_date == "2025-01-15"
            assert self.config["data_vendors"]["core_stock_apis"] == "maestro"
            return (
                {
                    "final_trade_decision": decision,
                    "market_report": "market",
                    "news_report": "news",
                    "fundamentals_report": "fundamentals",
                },
                decision,
            )

    monkeypatch.setattr(adapter, "TradingAgentsGraph", FakeGraph)
    result = TradingAgentsVirtuosoStrategy().run(DataBundle(data={}), _context())

    assert isinstance(result, TargetAllocationResult)
    assert result.allocations == {"AAPL": expected_weight, "CASH": 1.0 - expected_weight}
    assert result.confidence == expected_confidence
    assert result.metadata["raw_decision"] == decision
    assert result.metadata["rating"] in {"Buy", "Hold", "Sell"}


def test_data_bundle_vendor_formats_payloads_and_cleans_up(monkeypatch):
    original_methods = {
        method: dict(vendors)
        for method, vendors in dataflow_interface.VENDOR_METHODS.items()
    }
    original_config = dataflow_config.get_config()

    class FakeGraph:
        def __init__(self, selected_analysts, config):
            dataflow_config.set_config(config)

        def propagate(self, symbol, trade_date):
            stock_data = dataflow_interface.route_to_vendor(
                "get_stock_data", "AAPL", "2025-01-01", "2025-01-03"
            )
            news = dataflow_interface.route_to_vendor(
                "get_news", "AAPL", "2025-01-01", "2025-01-03"
            )
            fundamentals = dataflow_interface.route_to_vendor(
                "get_fundamentals", "AAPL", "2025-01-03"
            )
            indicator = dataflow_interface.route_to_vendor(
                "get_indicators", "AAPL", "sma_2", "2025-01-03", 3
            )
            assert "2025-01-02" in stock_data
            assert "Earnings beat" in news
            assert "market_cap: 100" in fundamentals
            assert "Total Assets" in dataflow_interface.route_to_vendor(
                "get_balance_sheet", "AAPL", "quarterly", "2025-01-03"
            )
            assert "Operating Cash Flow" in dataflow_interface.route_to_vendor(
                "get_cashflow", "AAPL", "quarterly", "2025-01-03"
            )
            assert "Revenue" in dataflow_interface.route_to_vendor(
                "get_income_statement", "AAPL", "quarterly", "2025-01-03"
            )
            assert "sma_2" in indicator
            return ({"final_trade_decision": "Rating: Overweight"}, "Rating: Overweight")

    monkeypatch.setattr(adapter, "TradingAgentsGraph", FakeGraph)
    bundle = DataBundle(
        data={
            "AAPL": {
                "ohlcv": {
                    "bars": [
                        {"date": "2025-01-01", "open": 1, "high": 2, "low": 1, "close": 1, "volume": 10},
                        {"date": "2025-01-02", "open": 2, "high": 3, "low": 2, "close": 2, "volume": 20},
                        {"date": "2025-01-03", "open": 3, "high": 4, "low": 3, "close": 3, "volume": 30},
                    ]
                },
                "news": {
                    "articles": [
                        {"date": "2025-01-02", "title": "Earnings beat", "summary": "Strong quarter"}
                    ]
                },
                "fundamental": {"market_cap": 100},
                "financial_statements": {
                    "balance_sheet": {
                        "statement": [{"period": "2024", "Total Assets": 1000}]
                    },
                    "cashflow": {
                        "statement": [{"period": "2024", "Operating Cash Flow": 120}]
                    },
                    "income_statement": {
                        "statement": [{"period": "2024", "Revenue": 500}]
                    },
                },
            },
            "MARKET": {"news": {"articles": [{"title": "Macro"}]}},
        }
    )

    TradingAgentsVirtuosoStrategy().run(bundle, _context())

    assert dataflow_interface.VENDOR_METHODS == original_methods
    assert dataflow_config.get_config() == original_config


def test_runtime_vendor_fetches_missing_payloads_during_graph(monkeypatch):
    class FakeRuntime:
        def __init__(self):
            self.requests = []

        def get_data(self, requests):
            self.requests.extend(requests)
            request = requests[0]
            if request.data_type == "ohlcv":
                return DataBundle(
                    data={
                        request.symbol: {
                            "bars": [
                                {"date": "2025-01-01", "open": 1, "high": 2, "low": 1, "close": 1, "volume": 10},
                                {"date": "2025-01-02", "open": 2, "high": 3, "low": 2, "close": 2, "volume": 20},
                            ]
                        }
                    }
                )
            if request.data_type == "technical_indicators":
                return DataBundle(
                    data={
                        request.symbol: {
                            "technical_indicators": {
                                request.indicator: {
                                    "values": [
                                        {"timestamp": "2025-01-02", "value": 1.5},
                                    ]
                                }
                            }
                        }
                    }
                )
            if request.data_type == "news":
                return DataBundle(
                    data={
                        request.symbol: {
                            "news": {
                                "articles": [
                                    {"date": "2025-01-02", "title": "Runtime news", "summary": "Fetched on demand"}
                                ]
                            }
                        }
                    }
                )
            if request.data_type == "financial_statements":
                return DataBundle(
                    data={
                        request.symbol: {
                            "financial_statements": {
                                request.statement_type: {
                                    "statement": [{"period": "2024", "Revenue": 500}]
                                }
                            }
                        }
                    }
                )
            return DataBundle(data={request.symbol: {}})

    class FakeGraph:
        def __init__(self, selected_analysts, config):
            dataflow_config.set_config(config)

        def propagate(self, symbol, trade_date):
            stock_data = dataflow_interface.route_to_vendor(
                "get_stock_data", "AAPL", "2025-01-01", "2025-01-03"
            )
            indicator = dataflow_interface.route_to_vendor(
                "get_indicators", "AAPL", "sma_2", "2025-01-03", 3
            )
            news = dataflow_interface.route_to_vendor(
                "get_news", "AAPL", "2025-01-01", "2025-01-03"
            )
            statement = dataflow_interface.route_to_vendor(
                "get_income_statement", "AAPL", "quarterly", "2025-01-03"
            )
            assert "2025-01-02" in stock_data
            assert "sma_2" in indicator
            assert "Runtime news" in news
            assert "Revenue" in statement
            return ({"final_trade_decision": "Rating: Hold"}, "Rating: Hold")

    monkeypatch.setattr(adapter, "TradingAgentsGraph", FakeGraph)
    runtime = FakeRuntime()

    TradingAgentsVirtuosoStrategy().run_with_runtime(DataBundle(data={}), _context(), runtime)

    assert [request.data_type for request in runtime.requests] == [
        "ohlcv",
        "technical_indicators",
        "news",
        "financial_statements",
    ]
    assert runtime.requests[1].indicator == "sma"
    assert runtime.requests[1].lookback == 2


def test_vendor_routing_is_restored_when_graph_raises(monkeypatch):
    original_methods = {
        method: dict(vendors)
        for method, vendors in dataflow_interface.VENDOR_METHODS.items()
    }
    original_config = dataflow_config.get_config()

    class FakeGraph:
        def __init__(self, selected_analysts, config):
            dataflow_config.set_config(config)

        def propagate(self, symbol, trade_date):
            raise RuntimeError("graph failed")

    monkeypatch.setattr(adapter, "TradingAgentsGraph", FakeGraph)

    with pytest.raises(RuntimeError, match="graph failed"):
        TradingAgentsVirtuosoStrategy().run(DataBundle(data={}), _context())

    assert dataflow_interface.VENDOR_METHODS == original_methods
    assert dataflow_config.get_config() == original_config


def test_adapter_imports_only_public_maestro_sdk():
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    maestro_imports = [name for name in imports if name == "maestro" or name.startswith("maestro.")]
    assert maestro_imports == ["maestro.sdk"]


def test_metadata_does_not_include_env_secret(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-key")

    class FakeGraph:
        def __init__(self, selected_analysts, config):
            pass

        def propagate(self, symbol, trade_date):
            return ({"final_trade_decision": "Rating: Hold"}, "Rating: Hold")

    monkeypatch.setattr(adapter, "TradingAgentsGraph", FakeGraph)

    result = TradingAgentsVirtuosoStrategy().run(DataBundle(data={}), _context())

    assert "super-secret-key" not in repr(result.metadata)
