"""Maestro Virtuoso adapter for TradingAgents."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

from maestro.sdk import (
    BaseStrategyPlugin,
    DataBundle,
    DataRequest,
    StrategyContext,
    StrategyManifest,
    StrategyRuntime,
    StrategySignalResult,
)
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.dataflows import config as dataflow_config
from tradingagents.dataflows import interface as dataflow_interface
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


UNAVAILABLE = "Maestro DataHub payload not supplied"


@dataclass(frozen=True)
class AdapterConfig:
    symbol: str
    asset_type: str
    cash_symbol: str
    selected_analysts: list[str] = field(
        default_factory=lambda: ["market", "news", "fundamentals"]
    )
    trade_date: str | None = None
    llm_provider: str = "openai"
    deep_think_llm: str = DEFAULT_CONFIG["deep_think_llm"]
    quick_think_llm: str = DEFAULT_CONFIG["quick_think_llm"]
    backend_url: str | None = DEFAULT_CONFIG["backend_url"]
    agent_llms: dict[str, Any] = field(default_factory=dict)
    output_language: str = DEFAULT_CONFIG["output_language"]
    max_debate_rounds: int = DEFAULT_CONFIG["max_debate_rounds"]
    max_risk_discuss_rounds: int = DEFAULT_CONFIG["max_risk_discuss_rounds"]
    ohlcv_lookback_days: int = 260
    news_lookback_days: int = 7
    news_limit: int = 20
    include_insider_transactions: bool = False


class TradingAgentsVirtuosoStrategy(BaseStrategyPlugin):
    """Maestro wrapper for the TradingAgents research graph."""

    VERSION = "0.1.0"

    def manifest(self) -> StrategyManifest:
        return StrategyManifest(
            sdk_contract_version="1.1",
            strategy_id="tradingagents",
            name="TradingAgents",
            version=self.VERSION,
            supported_modes=["paper", "live_approval"],
            supported_asset_types=["cash", "stock", "etf", "domestic_etf", "us_etf"],
            result_type="strategy_signal",
            requires_data=[
                "ohlcv",
                "price",
                "fundamental",
                "news",
                "financial_statements",
            ],
            requires_llm=True,
            supported_llm_providers=["openai", "openrouter", "anthropic", "google"],
            required_env_vars=[],
            can_run_live=True,
            allow_direct_external_data_calls=False,
            estimated_runtime_seconds=180,
        )

    def build_data_requests(self, context: StrategyContext) -> list[DataRequest]:
        cfg = _adapter_config(context)
        trade_date = _trade_date(cfg, context)
        requests = [
            _data_request(
                symbol=cfg.symbol,
                asset_type=cfg.asset_type,
                data_type="ohlcv",
                intended_use="tradable",
                timeframe="1d",
                lookback=cfg.ohlcv_lookback_days,
                as_of=trade_date,
            ),
            _data_request(
                symbol=cfg.symbol,
                asset_type=cfg.asset_type,
                data_type="price",
                intended_use="tradable",
                as_of=trade_date,
            ),
            _data_request(
                symbol=cfg.symbol,
                asset_type=cfg.asset_type,
                data_type="fundamental",
                intended_use="tradable",
                as_of=trade_date,
            ),
            _data_request(
                symbol=cfg.symbol,
                asset_type=cfg.asset_type,
                data_type="financial_statements",
                intended_use="research",
                statement_type="balance_sheet",
                frequency="annual",
                as_of=trade_date,
            ),
            _data_request(
                symbol=cfg.symbol,
                asset_type=cfg.asset_type,
                data_type="financial_statements",
                intended_use="research",
                statement_type="cashflow",
                frequency="annual",
                as_of=trade_date,
            ),
            _data_request(
                symbol=cfg.symbol,
                asset_type=cfg.asset_type,
                data_type="financial_statements",
                intended_use="research",
                statement_type="income_statement",
                frequency="annual",
                as_of=trade_date,
            ),
            _data_request(
                symbol=cfg.symbol,
                asset_type=cfg.asset_type,
                data_type="news",
                intended_use="tradable",
                lookback=cfg.news_lookback_days,
                limit=cfg.news_limit,
                as_of=trade_date,
            ),
            _data_request(
                symbol="MARKET",
                asset_type=cfg.asset_type,
                data_type="news",
                intended_use="research",
                lookback=cfg.news_lookback_days,
                limit=cfg.news_limit,
                query="market",
                as_of=trade_date,
            ),
        ]
        if cfg.include_insider_transactions:
            requests.append(
                _data_request(
                    symbol=cfg.symbol,
                    asset_type=cfg.asset_type,
                    data_type="insider_transactions",
                    intended_use="tradable",
                    as_of=trade_date,
                )
            )
        return requests

    def run(
        self,
        data_bundle: DataBundle,
        context: StrategyContext,
    ) -> StrategySignalResult:
        return self._run_with_bundle_view(_BundleView(data_bundle), context)

    def run_with_runtime(
        self,
        data_bundle: DataBundle,
        context: StrategyContext,
        runtime: StrategyRuntime,
    ) -> StrategySignalResult:
        cfg = _adapter_config(context)
        return self._run_with_bundle_view(
            _RuntimeBundleView(data_bundle, context, cfg, runtime),
            context,
        )

    def _run_with_bundle_view(
        self,
        bundle_view: "_BundleView",
        context: StrategyContext,
    ) -> StrategySignalResult:
        cfg = _adapter_config(context)
        run_mode = _run_mode_value(context)
        if run_mode not in {"paper", "live_approval"}:
            raise ValueError(
                "TradingAgents Virtuoso adapter supports paper and live_approval modes only"
            )

        trade_date = _trade_date(cfg, context)
        graph_config = _tradingagents_config(cfg)

        with _maestro_vendor(bundle_view):
            graph = TradingAgentsGraph(
                selected_analysts=cfg.selected_analysts,
                config=graph_config,
            )
            state, processed_signal = graph.propagate(cfg.symbol, trade_date)

        final_decision = _final_decision(state, processed_signal)
        rating = parse_rating(final_decision)
        confidence = {
            "Buy": 0.75,
            "Sell": 0.75,
            "Overweight": 0.65,
            "Underweight": 0.65,
            "Hold": 0.50,
        }[rating]

        return StrategySignalResult(
            strategy_id=_getattr(context, "strategy_id", self.manifest().strategy_id),
            strategy_version=self.manifest().version,
            timestamp=_getattr(context, "timestamp", datetime.utcnow()),
            symbol=cfg.symbol,
            action=_rating_to_action(rating),
            rating=rating,
            confidence=confidence,
            time_horizon="1-3 months",
            position_sizing="Maestro signal_to_allocation policy owns target weight",
            rationale=_summarize(final_decision),
            metadata={
                "rating": rating,
                "raw_decision": final_decision,
                "trade_date": trade_date,
                "llm_provider": cfg.llm_provider,
                "deep_think_llm": cfg.deep_think_llm,
                "quick_think_llm": cfg.quick_think_llm,
                "agent_llms": _safe_agent_llms(cfg.agent_llms),
                "reports_present": {
                    "market_report": bool(_mapping_get(state, "market_report")),
                    "sentiment_report": bool(_mapping_get(state, "sentiment_report")),
                    "news_report": bool(_mapping_get(state, "news_report")),
                    "fundamentals_report": bool(
                        _mapping_get(state, "fundamentals_report")
                    ),
                    "investment_plan": bool(_mapping_get(state, "investment_plan")),
                    "final_trade_decision": bool(final_decision),
                },
            },
        )


def _adapter_config(context: StrategyContext) -> AdapterConfig:
    raw = dict(_getattr(context, "config", {}) or {})
    missing = [
        key for key in ("symbol", "asset_type", "cash_symbol") if not raw.get(key)
    ]
    if missing:
        raise ValueError(
            f"Missing TradingAgents adapter config keys: {', '.join(missing)}"
        )

    defaults = AdapterConfig(
        symbol=str(raw["symbol"]),
        asset_type=str(raw["asset_type"]),
        cash_symbol=str(raw["cash_symbol"]),
    )
    values = {
        field_name: getattr(defaults, field_name)
        for field_name in defaults.__dataclass_fields__
    }
    values.update({key: raw[key] for key in values.keys() & raw.keys()})
    values["symbol"] = str(values["symbol"])
    values["asset_type"] = str(values["asset_type"])
    values["cash_symbol"] = str(values["cash_symbol"])
    values["selected_analysts"] = list(values["selected_analysts"])
    values["agent_llms"] = dict(values["agent_llms"] or {})
    return AdapterConfig(**values)


def _run_mode_value(context: StrategyContext) -> str:
    run_mode = _getattr(context, "run_mode", "paper")
    return str(getattr(run_mode, "value", run_mode))


def _rating_to_action(rating: str) -> str:
    if rating in {"Buy", "Overweight"}:
        return "buy"
    if rating in {"Underweight", "Sell"}:
        return "sell"
    return "hold"


def _safe_agent_llms(agent_llms: dict[str, Any]) -> dict[str, dict[str, Any]]:
    allowed_keys = {
        "provider",
        "model",
        "tier",
        "base_url",
        "backend_url",
        "reasoning_effort",
        "openai_reasoning_effort",
        "thinking_level",
        "google_thinking_level",
        "effort",
        "anthropic_effort",
        "timeout",
        "max_retries",
    }
    metadata = {}
    for agent_name, raw_spec in agent_llms.items():
        if isinstance(raw_spec, dict):
            metadata[str(agent_name)] = {
                str(key): value
                for key, value in raw_spec.items()
                if key in allowed_keys
            }
    return metadata


def _data_request(**kwargs: Any) -> DataRequest:
    return DataRequest(
        **{key: value for key, value in kwargs.items() if value is not None}
    )


def _runtime_data_request(
    *,
    cfg: AdapterConfig,
    symbol: str,
    data_type: str,
    intended_use: str = "research",
    **kwargs: Any,
) -> DataRequest:
    return _data_request(
        symbol=symbol,
        asset_type=cfg.asset_type,
        data_type=data_type,
        intended_use=intended_use,
        **kwargs,
    )


def _trade_date(cfg: AdapterConfig, context: StrategyContext) -> str:
    if cfg.trade_date:
        return cfg.trade_date
    timestamp = _getattr(context, "timestamp", None)
    if isinstance(timestamp, datetime):
        return timestamp.date().isoformat()
    if isinstance(timestamp, date):
        return timestamp.isoformat()
    if timestamp:
        return str(timestamp)[:10]
    return datetime.utcnow().date().isoformat()


def _datetime_or_none(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _tradingagents_config(cfg: AdapterConfig) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "llm_provider": cfg.llm_provider,
            "deep_think_llm": cfg.deep_think_llm,
            "quick_think_llm": cfg.quick_think_llm,
            "backend_url": cfg.backend_url,
            "agent_llms": _safe_agent_llms(cfg.agent_llms),
            "output_language": cfg.output_language,
            "max_debate_rounds": cfg.max_debate_rounds,
            "max_risk_discuss_rounds": cfg.max_risk_discuss_rounds,
            "checkpoint_enabled": False,
            "news_article_limit": cfg.news_limit,
            "global_news_article_limit": cfg.news_limit,
            "global_news_lookback_days": cfg.news_lookback_days,
            "data_vendors": {
                "core_stock_apis": "maestro",
                "technical_indicators": "maestro",
                "fundamental_data": "maestro",
                "news_data": "maestro",
            },
            "tool_vendors": {
                "get_stock_data": "maestro",
                "get_indicators": "maestro",
                "get_fundamentals": "maestro",
                "get_balance_sheet": "maestro",
                "get_cashflow": "maestro",
                "get_income_statement": "maestro",
                "get_news": "maestro",
                "get_global_news": "maestro",
                "get_insider_transactions": "maestro",
            },
        }
    )
    return config


@contextmanager
def _maestro_vendor(bundle: "_BundleView"):
    original_methods = deepcopy(dataflow_interface.VENDOR_METHODS)
    original_config = deepcopy(dataflow_config._config)
    dataflow_interface.VENDOR_METHODS["get_stock_data"]["maestro"] = (
        lambda symbol, start_date, end_date: _format_ohlcv(
            bundle.stock_payload(symbol, start_date, end_date), start_date, end_date
        )
    )
    dataflow_interface.VENDOR_METHODS["get_indicators"]["maestro"] = (
        lambda symbol, indicator, curr_date, look_back_days=30: (
            _format_runtime_indicator(
                bundle.indicator_payload(symbol, indicator, curr_date, look_back_days),
                indicator,
            )
            or _format_indicator(
                bundle.indicator_ohlcv_payload(
                    symbol, indicator, curr_date, look_back_days
                ),
                indicator,
                curr_date,
                look_back_days,
            )
        )
    )
    dataflow_interface.VENDOR_METHODS["get_fundamentals"]["maestro"] = (
        lambda ticker, curr_date: _format_mapping(
            bundle.fundamental_payload(ticker, curr_date), "fundamental"
        )
    )
    dataflow_interface.VENDOR_METHODS["get_balance_sheet"]["maestro"] = (
        lambda ticker, freq="quarterly", curr_date=None: _format_statement(
            bundle.statement_payload(ticker, "balance_sheet", freq, curr_date),
            "balance_sheet",
        )
    )
    dataflow_interface.VENDOR_METHODS["get_cashflow"]["maestro"] = (
        lambda ticker, freq="quarterly", curr_date=None: _format_statement(
            bundle.statement_payload(ticker, "cashflow", freq, curr_date),
            "cashflow",
        )
    )
    dataflow_interface.VENDOR_METHODS["get_income_statement"]["maestro"] = (
        lambda ticker, freq="quarterly", curr_date=None: _format_statement(
            bundle.statement_payload(ticker, "income_statement", freq, curr_date),
            "income_statement",
        )
    )
    dataflow_interface.VENDOR_METHODS["get_news"]["maestro"] = (
        lambda ticker, start_date, end_date: _format_news(
            bundle.news_payload(ticker, start_date, end_date)
        )
    )
    dataflow_interface.VENDOR_METHODS["get_global_news"]["maestro"] = (
        lambda curr_date, look_back_days=None, limit=None: _format_news(
            bundle.global_news_payload(curr_date, look_back_days, limit), limit=limit
        )
    )
    dataflow_interface.VENDOR_METHODS["get_insider_transactions"]["maestro"] = (
        lambda ticker: _format_mapping(
            bundle.insider_transactions_payload(ticker), "insider_transactions"
        )
    )
    try:
        yield
    finally:
        dataflow_interface.VENDOR_METHODS.clear()
        dataflow_interface.VENDOR_METHODS.update(original_methods)
        dataflow_config._config = original_config


class _BundleView:
    def __init__(self, data_bundle: DataBundle):
        self.data = _getattr(data_bundle, "data", data_bundle)

    def payload(self, symbol: str, data_type: str) -> Any:
        found = self._find(self.data, symbol, data_type)
        return _unwrap_payload(found)

    def stock_payload(
        self, symbol: str, start_date: str | None, end_date: str | None
    ) -> Any:
        del start_date, end_date
        return self.payload(symbol, "ohlcv")

    def indicator_payload(
        self,
        symbol: str,
        indicator: str,
        curr_date: str | None,
        look_back_days: int = 30,
    ) -> Any:
        del symbol, indicator, curr_date, look_back_days
        return None

    def indicator_ohlcv_payload(
        self,
        symbol: str,
        indicator: str,
        curr_date: str | None,
        look_back_days: int = 30,
    ) -> Any:
        del indicator, curr_date, look_back_days
        return self.payload(symbol, "ohlcv")

    def fundamental_payload(self, ticker: str, curr_date: str | None) -> Any:
        del curr_date
        return self.payload(ticker, "fundamental")

    def statement_payload(
        self,
        ticker: str,
        statement_type: str,
        frequency: str | None,
        curr_date: str | None,
    ) -> Any:
        del frequency, curr_date
        return self.payload(ticker, statement_type) or self.payload(
            ticker, "financial_statements"
        )

    def news_payload(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str | None,
    ) -> Any:
        del start_date, end_date
        return self.payload(ticker, "news")

    def global_news_payload(
        self,
        curr_date: str | None,
        look_back_days: int | None,
        limit: int | None,
    ) -> Any:
        del curr_date, look_back_days, limit
        return self.payload("MARKET", "news")

    def insider_transactions_payload(self, ticker: str) -> Any:
        return self.payload(ticker, "insider_transactions")

    def _find(self, data: Any, symbol: str, data_type: str) -> Any:
        if data is None:
            return None
        if isinstance(data, list):
            for item in data:
                item_symbol = _mapping_get(item, "symbol")
                item_type = _mapping_get(item, "data_type") or _mapping_get(
                    item, "type"
                )
                if item_symbol == symbol and item_type == data_type:
                    return item
            return None
        if not isinstance(data, dict):
            return None

        for key in (
            (symbol, data_type),
            f"{symbol}:{data_type}",
            f"{symbol}.{data_type}",
        ):
            if key in data:
                return data[key]

        by_symbol = data.get(symbol)
        if isinstance(by_symbol, dict) and data_type in by_symbol:
            return by_symbol[data_type]
        if isinstance(by_symbol, dict) and _payload_matches_data_type(
            by_symbol, data_type
        ):
            return by_symbol

        by_type = data.get(data_type)
        if isinstance(by_type, dict) and symbol in by_type:
            return by_type[symbol]

        return None


class _RuntimeBundleView(_BundleView):
    def __init__(
        self,
        data_bundle: DataBundle,
        context: StrategyContext,
        cfg: AdapterConfig,
        runtime: StrategyRuntime,
    ) -> None:
        super().__init__(data_bundle)
        self.context = context
        self.cfg = cfg
        self.runtime = runtime
        self._runtime_cache: dict[tuple[Any, ...], Any] = {}

    def stock_payload(
        self, symbol: str, start_date: str | None, end_date: str | None
    ) -> Any:
        prefetched = self.payload(symbol, "ohlcv")
        if prefetched is not None:
            return prefetched
        return self._runtime_payload(
            (
                "stock",
                symbol,
                start_date,
                end_date,
            ),
            _runtime_data_request(
                cfg=self.cfg,
                symbol=symbol,
                data_type="ohlcv",
                intended_use=self._intended_use(symbol),
                start=_datetime_or_none(start_date),
                end=_datetime_or_none(end_date),
                as_of=_datetime_or_none(end_date),
            ),
            symbol,
            "ohlcv",
        )

    def indicator_payload(
        self,
        symbol: str,
        indicator: str,
        curr_date: str | None,
        look_back_days: int = 30,
    ) -> Any:
        base_indicator, window = _indicator_request_parts(indicator, look_back_days)
        return self._runtime_payload(
            (
                "indicator",
                symbol,
                base_indicator,
                window,
                curr_date,
            ),
            _runtime_data_request(
                cfg=self.cfg,
                symbol=symbol,
                data_type="technical_indicators",
                intended_use=self._intended_use(symbol),
                indicator=base_indicator,
                lookback=window,
                as_of=_datetime_or_none(curr_date),
            ),
            symbol,
            "technical_indicators",
        )

    def indicator_ohlcv_payload(
        self,
        symbol: str,
        indicator: str,
        curr_date: str | None,
        look_back_days: int = 30,
    ) -> Any:
        prefetched = self.payload(symbol, "ohlcv")
        if prefetched is not None:
            return prefetched
        return self.stock_payload(symbol, None, curr_date)

    def fundamental_payload(self, ticker: str, curr_date: str | None) -> Any:
        prefetched = self.payload(ticker, "fundamental")
        if prefetched is not None:
            return prefetched
        return self._runtime_payload(
            ("fundamental", ticker, curr_date),
            _runtime_data_request(
                cfg=self.cfg,
                symbol=ticker,
                data_type="fundamental",
                intended_use=self._intended_use(ticker),
                as_of=_datetime_or_none(curr_date),
            ),
            ticker,
            "fundamental",
        )

    def statement_payload(
        self,
        ticker: str,
        statement_type: str,
        frequency: str | None,
        curr_date: str | None,
    ) -> Any:
        prefetched = self.payload(ticker, statement_type) or self.payload(
            ticker, "financial_statements"
        )
        if prefetched is not None:
            return prefetched
        return self._runtime_payload(
            ("statement", ticker, statement_type, frequency, curr_date),
            _runtime_data_request(
                cfg=self.cfg,
                symbol=ticker,
                data_type="financial_statements",
                intended_use="research",
                statement_type=statement_type,
                frequency=_statement_frequency(frequency),
                as_of=_datetime_or_none(curr_date),
            ),
            ticker,
            "financial_statements",
        )

    def news_payload(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str | None,
    ) -> Any:
        prefetched = self.payload(ticker, "news")
        if prefetched is not None:
            return prefetched
        return self._runtime_payload(
            ("news", ticker, start_date, end_date),
            _runtime_data_request(
                cfg=self.cfg,
                symbol=ticker,
                data_type="news",
                intended_use=self._intended_use(ticker),
                start=_datetime_or_none(start_date),
                end=_datetime_or_none(end_date),
                as_of=_datetime_or_none(end_date),
            ),
            ticker,
            "news",
        )

    def global_news_payload(
        self,
        curr_date: str | None,
        look_back_days: int | None,
        limit: int | None,
    ) -> Any:
        prefetched = self.payload("MARKET", "news")
        if prefetched is not None:
            return prefetched
        return self._runtime_payload(
            ("global_news", curr_date, look_back_days, limit),
            _runtime_data_request(
                cfg=self.cfg,
                symbol="MARKET",
                data_type="news",
                intended_use="research",
                lookback=look_back_days,
                limit=limit,
                query="market",
                as_of=_datetime_or_none(curr_date),
            ),
            "MARKET",
            "news",
        )

    def insider_transactions_payload(self, ticker: str) -> Any:
        prefetched = self.payload(ticker, "insider_transactions")
        if prefetched is not None:
            return prefetched
        return self._runtime_payload(
            ("insider_transactions", ticker),
            _runtime_data_request(
                cfg=self.cfg,
                symbol=ticker,
                data_type="insider_transactions",
                intended_use=self._intended_use(ticker),
                as_of=_datetime_or_none(_trade_date(self.cfg, self.context)),
            ),
            ticker,
            "insider_transactions",
        )

    def _runtime_payload(
        self,
        cache_key: tuple[Any, ...],
        request: DataRequest,
        symbol: str,
        data_type: str,
    ) -> Any:
        if cache_key in self._runtime_cache:
            return self._runtime_cache[cache_key]
        try:
            bundle = self.runtime.get_data([request])
        except Exception:
            self._runtime_cache[cache_key] = None
            return None
        payload = _BundleView(bundle).payload(symbol, data_type)
        self._runtime_cache[cache_key] = payload
        return payload

    def _intended_use(self, symbol: str) -> str:
        return "research" if symbol == "MARKET" else "tradable"


def _unwrap_payload(value: Any) -> Any:
    if value is None:
        return None
    for attr in ("payload", "data", "value"):
        nested = _getattr(value, attr, None)
        if nested is not None and nested is not value:
            return nested
    if isinstance(value, dict):
        for key in ("payload", "data", "value"):
            if key in value:
                return value[key]
    return value


def _payload_matches_data_type(payload: dict[str, Any], data_type: str) -> bool:
    markers = {
        "ohlcv": ("bars", "ohlcv", "prices"),
        "price": ("latest_price", "price"),
        "fundamental": ("fundamental", "metrics", "raw_info"),
        "news": ("articles", "news", "items", "latest"),
        "technical_indicators": ("technical_indicators", "indicator", "values"),
        "financial_statements": ("financial_statements",),
        "balance_sheet": ("balance_sheet",),
        "cashflow": ("cashflow",),
        "income_statement": ("income_statement",),
        "insider_transactions": ("insider_transactions",),
    }
    return any(marker in payload for marker in markers.get(data_type, ()))


def _indicator_request_parts(
    indicator: str, look_back_days: int | None
) -> tuple[str, int | None]:
    name = str(indicator).strip().lower()
    window = _window_from_name(name, 0) or look_back_days
    if name.startswith("macd"):
        return "macd", None
    if name.startswith("rsi"):
        return "rsi", window or 14
    if name.startswith("ema"):
        return "ema", window or 20
    if name.startswith("sma") or name.startswith("ma"):
        return "sma", window or 20
    if name.startswith("boll") or name.startswith("bbands"):
        return "bollinger", window or 20
    return name, window


def _statement_frequency(frequency: str | None) -> str:
    normalized = str(frequency or "annual").strip().lower()
    return normalized if normalized in {"annual", "quarterly", "trailing"} else "annual"


def _format_ohlcv(
    payload: Any,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    frame = _bars_frame(payload)
    if frame.empty:
        return f"{UNAVAILABLE}: ohlcv"
    if "date" in frame.columns:
        frame = frame.sort_values("date")
        if start_date:
            frame = frame[frame["date"].astype(str) >= str(start_date)]
        if end_date:
            frame = frame[frame["date"].astype(str) <= str(end_date)]
    return frame.to_csv(index=False)


def _format_indicator(
    payload: Any,
    indicator: str,
    curr_date: str | None,
    look_back_days: int = 30,
) -> str:
    frame = _bars_frame(payload)
    if frame.empty or "close" not in frame.columns:
        return f"{UNAVAILABLE}: ohlcv for technical indicator {indicator}"
    frame = frame.sort_values("date") if "date" in frame.columns else frame
    if curr_date and "date" in frame.columns:
        frame = frame[frame["date"].astype(str) <= str(curr_date)]
    frame = frame.tail(max(int(look_back_days), 1))
    name = indicator.strip().lower()
    result = pd.DataFrame(
        {"date": frame["date"] if "date" in frame.columns else range(len(frame))}
    )

    close = pd.to_numeric(frame["close"], errors="coerce")
    if name.startswith("rsi"):
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        result[name] = 100 - (100 / (1 + gain / loss.replace(0, pd.NA)))
    elif name.startswith("macd"):
        macd = (
            close.ewm(span=12, adjust=False).mean()
            - close.ewm(span=26, adjust=False).mean()
        )
        result["macd"] = macd
        result["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
        result["macd_hist"] = result["macd"] - result["macd_signal"]
    elif name.startswith("ema"):
        result[name] = close.ewm(span=_window_from_name(name, 20), adjust=False).mean()
    elif name.startswith("sma") or name.startswith("ma"):
        window = _window_from_name(name, 20)
        result[name] = close.rolling(window, min_periods=1).mean()
    elif name.startswith("boll"):
        middle = close.rolling(20, min_periods=1).mean()
        std = close.rolling(20, min_periods=1).std().fillna(0)
        result["bollinger_middle"] = middle
        result["bollinger_upper"] = middle + 2 * std
        result["bollinger_lower"] = middle - 2 * std
    else:
        return f"{UNAVAILABLE}: unsupported technical indicator {indicator}"

    return result.to_csv(index=False)


def _format_runtime_indicator(payload: Any, indicator: str) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict) and "technical_indicators" in payload:
        indicators = payload.get("technical_indicators")
        if isinstance(indicators, dict):
            base_indicator, _ = _indicator_request_parts(indicator, None)
            payload = indicators.get(base_indicator) or next(
                iter(indicators.values()), None
            )
    elif isinstance(payload, dict) and "values" not in payload:
        base_indicator, _ = _indicator_request_parts(indicator, None)
        payload = payload.get(base_indicator) or next(iter(payload.values()), None)
    if isinstance(payload, dict) and "values" in payload:
        payload = payload["values"]
    if not isinstance(payload, list) or not payload:
        return None

    frame = pd.DataFrame(payload)
    if frame.empty:
        return None
    frame = frame.rename(columns={"timestamp": "date", "signal": "macd_signal"})
    if "value" in frame.columns:
        frame = frame.rename(columns={"value": str(indicator).strip().lower()})
    if "histogram" in frame.columns:
        frame = frame.rename(columns={"histogram": "macd_hist"})
    if {"middle", "upper", "lower"} <= set(frame.columns):
        frame = frame.rename(
            columns={
                "middle": "bollinger_middle",
                "upper": "bollinger_upper",
                "lower": "bollinger_lower",
            }
        )
    return frame.to_csv(index=False)


def _window_from_name(name: str, default: int) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else default


def _format_news(payload: Any, limit: int | None = None) -> str:
    items = _items(payload, ("articles", "news", "items", "data"))
    if not items:
        return f"{UNAVAILABLE}: news"
    rows = items[:limit] if limit else items
    formatted = []
    for item in rows:
        title = (
            _mapping_get(item, "title") or _mapping_get(item, "headline") or "Untitled"
        )
        published = (
            _mapping_get(item, "published_at") or _mapping_get(item, "date") or ""
        )
        source = _mapping_get(item, "source") or ""
        summary = (
            _mapping_get(item, "summary") or _mapping_get(item, "description") or ""
        )
        url = _mapping_get(item, "url") or ""
        formatted.append(
            f"- {published} {title}\n  Source: {source}\n  Summary: {summary}\n  URL: {url}"
        )
    return "\n".join(formatted)


def _format_statement(payload: Any, statement_type: str) -> str:
    if isinstance(payload, dict) and statement_type in payload:
        payload = payload[statement_type]
    return _format_mapping(payload, statement_type)


def _format_mapping(payload: Any, label: str) -> str:
    if isinstance(payload, pd.DataFrame):
        if payload.empty:
            return f"{UNAVAILABLE}: {label}"
        return payload.to_csv(index=False)
    if payload is None or payload == [] or payload == {}:
        return f"{UNAVAILABLE}: {label}"
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        return (
            pd.DataFrame(payload).to_csv(index=False)
            if payload
            else f"{UNAVAILABLE}: {label}"
        )
    if isinstance(payload, dict):
        return "\n".join(f"{key}: {value}" for key, value in payload.items())
    return str(payload)


def _bars_frame(payload: Any) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        rows = payload
        if isinstance(payload, dict):
            for key in ("bars", "ohlcv", "prices", "data"):
                if key in payload:
                    rows = payload[key]
                    break
        frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame.columns = [
        str(column).strip().lower().replace(" ", "_") for column in frame.columns
    ]
    rename = {
        "datetime": "date",
        "timestamp": "date",
        "adj_close": "adjusted_close",
    }
    frame = frame.rename(
        columns={key: value for key, value in rename.items() if key in frame.columns}
    )
    return frame


def _items(payload: Any, keys: Iterable[str]) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _final_decision(state: Any, processed_signal: Any) -> str:
    decision = _mapping_get(state, "final_trade_decision")
    if decision:
        return str(decision)
    if processed_signal:
        return str(processed_signal)
    return "Rating: Hold"


def _summarize(text: str, max_chars: int = 700) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= max_chars else compact[: max_chars - 3] + "..."


def _mapping_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _getattr(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
