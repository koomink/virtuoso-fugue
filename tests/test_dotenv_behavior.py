from __future__ import annotations

import importlib
import sys


def test_import_tradingagents_does_not_load_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    sys.modules.pop("tradingagents", None)

    importlib.import_module("tradingagents")

    assert "OPENAI_API_KEY" not in sys.modules["os"].environ

