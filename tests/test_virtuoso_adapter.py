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
class StrategySignalResult:
    strategy_id: str
    strategy_version: str
    timestamp: datetime
    symbol: str
    action: str
    rating: str | None
    confidence: float
    time_horizon: str | None = None
    position_sizing: str | None = None
    rationale: str | None = None
    risk_flags: list[str] | None = None
    metadata: dict | None = None


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
sdk.StrategySignalResult = StrategySignalResult

maestro = types.ModuleType("maestro")
maestro.sdk = sdk
sys.modules.setdefault("maestro", maestro)
sys.modules.setdefault("maestro.sdk", sdk)

from tradingagents.dataflows import config as dataflow_config  # noqa: E402
from tradingagents.dataflows import interface as dataflow_interface  # noqa: E402
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph import trading_graph as graph_module  # noqa: E402
from tradingagents_virtuoso import strategy as adapter  # noqa: E402
from tradingagents_virtuoso.strategy import TradingAgentsVirtuosoStrategy  # noqa: E402


def _context(**overrides):
    context_fields = {}
    for key in ("run_mode", "strategy_id", "timestamp"):
        if key in overrides:
            context_fields[key] = overrides.pop(key)
    config = {
        "symbol": "AAPL",
        "asset_type": "stock",
        "cash_symbol": "CASH",
    }
    config.update(overrides)
    return StrategyContext(config=config, **context_fields)


def test_manifest_matches_maestro_contract():
    plugin = TradingAgentsVirtuosoStrategy()

    manifest = plugin.manifest()

    assert isinstance(plugin, BaseStrategyPlugin)
    assert manifest.strategy_id == "tradingagents"
    assert manifest.sdk_contract_version == "1.1"
    assert manifest.result_type == "strategy_signal"
    assert manifest.supported_modes == ["paper", "live_approval"]
    assert "financial_statements" in manifest.requires_data
    assert manifest.requires_llm is True
    assert "openrouter" in manifest.supported_llm_providers
    assert manifest.required_env_vars == []
    assert manifest.can_run_live is True
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
        req
        for req in requests
        if req.symbol == "AAPL" and req.data_type == "financial_statements"
    ]
    assert {req.statement_type for req in statement_requests} == {
        "balance_sheet",
        "cashflow",
        "income_statement",
    }
    assert {req.frequency for req in statement_requests} == {"annual"}


@pytest.mark.parametrize(
    ("decision", "expected_action", "expected_rating", "expected_confidence"),
    [
        ("Rating: Buy\nAdd exposure.", "buy", "Buy", 0.75),
        ("Rating: Overweight\nLean positive.", "buy", "Overweight", 0.65),
        ("Rating: Hold\nWait for clarity.", "hold", "Hold", 0.50),
        ("Rating: Underweight\nReduce exposure.", "sell", "Underweight", 0.65),
        ("Rating: Sell\nExit risk.", "sell", "Sell", 0.75),
    ],
)
def test_run_maps_tradingagents_rating_to_strategy_signal(
    monkeypatch, decision, expected_action, expected_rating, expected_confidence
):
    class FakeGraph:
        def __init__(self, selected_analysts, config):
            self.selected_analysts = selected_analysts
            self.config = config
            assert selected_analysts == ["market", "news", "fundamentals"]

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

    assert isinstance(result, StrategySignalResult)
    assert result.symbol == "AAPL"
    assert result.action == expected_action
    assert result.rating == expected_rating
    assert result.confidence == expected_confidence
    assert result.metadata["raw_decision"] == decision
    assert result.metadata["rating"] == expected_rating
    assert (
        result.position_sizing
        == "Maestro signal_to_allocation policy owns target weight"
    )


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
            return (
                {"final_trade_decision": "Rating: Overweight"},
                "Rating: Overweight",
            )

    monkeypatch.setattr(adapter, "TradingAgentsGraph", FakeGraph)
    bundle = DataBundle(
        data={
            "AAPL": {
                "ohlcv": {
                    "bars": [
                        {
                            "date": "2025-01-01",
                            "open": 1,
                            "high": 2,
                            "low": 1,
                            "close": 1,
                            "volume": 10,
                        },
                        {
                            "date": "2025-01-02",
                            "open": 2,
                            "high": 3,
                            "low": 2,
                            "close": 2,
                            "volume": 20,
                        },
                        {
                            "date": "2025-01-03",
                            "open": 3,
                            "high": 4,
                            "low": 3,
                            "close": 3,
                            "volume": 30,
                        },
                    ]
                },
                "news": {
                    "articles": [
                        {
                            "date": "2025-01-02",
                            "title": "Earnings beat",
                            "summary": "Strong quarter",
                        }
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


def test_run_accepts_live_approval_mode(monkeypatch):
    class FakeGraph:
        def __init__(self, selected_analysts, config):
            pass

        def propagate(self, symbol, trade_date):
            return ({"final_trade_decision": "Rating: Buy"}, "Rating: Buy")

    monkeypatch.setattr(adapter, "TradingAgentsGraph", FakeGraph)

    result = TradingAgentsVirtuosoStrategy().run(
        DataBundle(data={}),
        _context(run_mode="live_approval"),
    )

    assert result.action == "buy"
    assert result.rating == "Buy"


def test_run_forwards_openrouter_and_agent_llm_overrides(monkeypatch):
    class FakeGraph:
        def __init__(self, selected_analysts, config):
            assert config["llm_provider"] == "openrouter"
            assert config["quick_think_llm"] == "openai/gpt-4o-mini"
            assert config["deep_think_llm"] == "anthropic/claude-sonnet-4.5"
            assert config["backend_url"] is None
            assert config["agent_llms"] == {
                "market": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o-mini",
                },
                "portfolio_manager": {
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                },
            }

        def propagate(self, symbol, trade_date):
            return ({"final_trade_decision": "Rating: Hold"}, "Rating: Hold")

    monkeypatch.setattr(adapter, "TradingAgentsGraph", FakeGraph)

    result = TradingAgentsVirtuosoStrategy().run(
        DataBundle(data={}),
        _context(
            llm_provider="openrouter",
            quick_think_llm="openai/gpt-4o-mini",
            deep_think_llm="anthropic/claude-sonnet-4.5",
            agent_llms={
                "market": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o-mini",
                    "api_key": "must-not-pass-through",
                },
                "portfolio_manager": {
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                },
            },
        ),
    )

    assert result.action == "hold"
    assert "must-not-pass-through" not in repr(result.metadata)
    assert result.metadata["llm_provider"] == "openrouter"
    assert result.metadata["agent_llms"]["market"] == {
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini",
    }


def test_run_rejects_unsupported_mode(monkeypatch):
    class FakeGraph:
        def __init__(self, selected_analysts, config):
            pass

        def propagate(self, symbol, trade_date):
            return ({"final_trade_decision": "Rating: Hold"}, "Rating: Hold")

    monkeypatch.setattr(adapter, "TradingAgentsGraph", FakeGraph)

    with pytest.raises(ValueError, match="paper and live_approval"):
        TradingAgentsVirtuosoStrategy().run(
            DataBundle(data={}),
            _context(run_mode="live_readonly"),
        )


def test_tradingagents_graph_builds_per_agent_llm_overrides(monkeypatch, tmp_path):
    created = []

    class FakeClient:
        def __init__(self, provider, model, base_url, kwargs):
            self.provider = provider
            self.model = model
            self.base_url = base_url
            self.kwargs = kwargs

        def get_llm(self):
            return FakeLLM(self.provider, self.model)

    class FakeLLM:
        def __init__(self, provider, model):
            self.provider = provider
            self.model = model

        @property
        def label(self):
            return f"{self.provider}:{self.model}"

        def with_structured_output(self, schema):
            del schema
            return self

        def bind_tools(self, tools):
            del tools
            return self

    def fake_create_llm_client(provider, model, base_url=None, **kwargs):
        created.append(
            {
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "kwargs": kwargs,
            }
        )
        return FakeClient(provider, model, base_url, kwargs)

    monkeypatch.setattr(graph_module, "create_llm_client", fake_create_llm_client)
    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            "project_dir": str(tmp_path),
            "results_dir": str(tmp_path / "results"),
            "data_cache_dir": str(tmp_path / "cache"),
            "memory_log_path": str(tmp_path / "memory.md"),
            "llm_provider": "openai",
            "quick_think_llm": "gpt-5.4-mini",
            "deep_think_llm": "gpt-5.4",
            "agent_llms": {
                "market": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o-mini",
                },
                "portfolio_manager": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "effort": "high",
                },
            },
        }
    )

    graph = graph_module.TradingAgentsGraph(
        selected_analysts=["market"],
        config=config,
    )

    assert graph.graph_setup.agent_llms["market"].label == (
        "openrouter:openai/gpt-4o-mini"
    )
    assert graph.graph_setup.agent_llms["portfolio_manager"].label == (
        "anthropic:claude-sonnet-4-5"
    )
    assert {(item["provider"], item["model"]) for item in created} >= {
        ("openai", "gpt-5.4"),
        ("openai", "gpt-5.4-mini"),
        ("openrouter", "openai/gpt-4o-mini"),
        ("anthropic", "claude-sonnet-4-5"),
    }
    assert created[-1]["kwargs"]["effort"] == "high"


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
                                {
                                    "date": "2025-01-01",
                                    "open": 1,
                                    "high": 2,
                                    "low": 1,
                                    "close": 1,
                                    "volume": 10,
                                },
                                {
                                    "date": "2025-01-02",
                                    "open": 2,
                                    "high": 3,
                                    "low": 2,
                                    "close": 2,
                                    "volume": 20,
                                },
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
                                    {
                                        "date": "2025-01-02",
                                        "title": "Runtime news",
                                        "summary": "Fetched on demand",
                                    }
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

    TradingAgentsVirtuosoStrategy().run_with_runtime(
        DataBundle(data={}), _context(), runtime
    )

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

    maestro_imports = [
        name for name in imports if name == "maestro" or name.startswith("maestro.")
    ]
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
