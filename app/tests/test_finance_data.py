# tests/test_finance_data.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from app.utils.finance_data import FinanceDataManager, calculate_growth_rate

@pytest.fixture
def manager(tmp_path):
    """A manager instance with a real temp directory — no Flask needed."""
    return FinanceDataManager(cache_dir=str(tmp_path), category_name="stocks")


# --- Pure logic tests (no mocking needed) ---

def test_calculate_growth_rate_positive():
    idx = pd.to_datetime(["2019-01-01","2020-01-01","2021-01-01","2022-01-01"])
    divs = pd.Series([1.0, 1.1, 1.21, 1.331], index=idx)
    rate = calculate_growth_rate(divs)
    assert abs(rate - 0.10) < 0.01   # ~10% CAGR

def test_calculate_growth_rate_insufficient_data():
    idx = pd.to_datetime(["2022-01-01"])
    divs = pd.Series([1.0], index=idx)
    assert calculate_growth_rate(divs) == "N/A"


# --- I/O helper tests ---

def test_load_json_missing_file(manager):
    result = manager._load_json("/nonexistent/path.json", default={"a": 1})
    assert result == {"a": 1}

def test_load_json_corrupt_file(manager, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json{{{")
    result = manager._load_json(str(bad), default={"fallback": True})
    assert result == {"fallback": True}

def test_save_and_load_json_roundtrip(manager, tmp_path):
    path = str(tmp_path / "test.json")
    manager._save_json(path, {"key": "value"})
    assert manager._load_json(path, default={}) == {"key": "value"}

def test_normalize_tz_strips_timezone(manager):
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame({"A": [1, 2, 3]}, index=idx)
    result = manager._normalize_tz(df)
    assert result.index.tz is None

def test_normalize_tz_passthrough_when_naive(manager):
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"A": [1, 2, 3]}, index=idx)
    result = manager._normalize_tz(df)
    assert result.index.tz is None  # still naive, no error


# --- Staleness logic tests (no network needed) ---

def test_is_stale_empty_df(manager):
    assert manager._is_stale(pd.DataFrame(), "1d") is True

def test_is_stale_fresh_data(manager):
    idx = pd.DatetimeIndex([datetime.now() - timedelta(minutes=10)])
    df = pd.DataFrame({"A": [1]}, index=idx)
    assert manager._is_stale(df, "4h") is False

def test_is_stale_old_data(manager):
    idx = pd.DatetimeIndex([datetime.now() - timedelta(hours=3)])
    df = pd.DataFrame({"A": [1]}, index=idx)
    assert manager._is_stale(df, "4h") is True


# --- Worker tests (mock yfinance) ---

def test_fill_valuation_fallback_pe(manager):
    manager._usd_eur = 0.92
    manager._chf_eur = 1.05
    data = {"Quote": 100.0, "Quote_EUR": 92.0, "P/E": 0.0, "Fwd_P/E": 0.0,
            "P/B": 0.0, "PEG": 0.0, "Currency": "USD", "Sector": "N/A",
            "PayoutRatio": 0.0}
    info = {"trailingPE": None, "forwardPE": 20.0, "priceToBook": 2.0,
            "payoutRatio": 0.4, "sector": "Technology",
            "trailingEps": 5.0, "currentPrice": 100.0, "currency": "USD"}
    manager._fill_valuation(data, info, "USD", "TEST")
    assert data["P/E"] == 20.0   # 100 / 5
    assert data["Fwd_P/E"] == 20.0