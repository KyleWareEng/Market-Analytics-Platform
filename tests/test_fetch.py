"""
Unit tests for data ingestion module.

Uses unittest.mock to simulate Yahoo Finance API and SQLite database interactions.
# NOTE: We use mocking here to avoid hitting the actual external API during testing.
# This ensures tests are fast, deterministic, and don't get rate-limited.
Run: python -m pytest tests/ -v
"""

import sqlite3
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, ANY
from datetime import datetime, timedelta

from src.fetch import fetch_price_data, _is_cache_valid, _read_from_cache


@pytest.fixture
def mock_db():
    """Create an in-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    
    conn.execute("""
        CREATE TABLE price_cache (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            fetched_at TEXT,
            period_fetched TEXT,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        CREATE TABLE cache_metadata (
            ticker TEXT PRIMARY KEY,
            last_fetched TEXT,
            period TEXT
        )
    """)
    yield conn
    conn.close()


@pytest.fixture
def mock_yf_data():
    """Create a sample DataFrame mimicking yfinance output."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    df = pd.DataFrame({
        "Open": 100.0,
        "High": 105.0,
        "Low": 95.0,
        "Close": 101.0,
        "Volume": 1000,
    }, index=dates)
    df.index.name = "Date"
    return df


def test_cache_logic_validity(mock_db):
    """Test _is_cache_valid logic."""
    ticker = "TEST"
    now = datetime.now()
    
    # Insert fresh data
    mock_db.execute(
        "INSERT INTO cache_metadata (ticker, last_fetched, period) VALUES (?, ?, ?)",
        (ticker, now.isoformat(), "1y")
    )
    
    # Should be valid for same period
    assert _is_cache_valid(mock_db, ticker, "1y") is True
    # Should be valid for shorter period (subset) - assuming naive check in logic
    # In my logic: cached "1y" (365 days) > requested "3mo" (90 days) -> True
    assert _is_cache_valid(mock_db, ticker, "3mo") is True
    
    # Should be invalid for longer period
    assert _is_cache_valid(mock_db, ticker, "2y") is False


def test_cache_expiry(mock_db):
    """Test cache expiration."""
    ticker = "TEST"
    # Insert old data (yesterday)
    old_time = (datetime.now() - timedelta(hours=24)).isoformat()
    mock_db.execute(
        "INSERT INTO cache_metadata (ticker, last_fetched, period) VALUES (?, ?, ?)",
        (ticker, old_time, "1y")
    )
    assert _is_cache_valid(mock_db, ticker, "1y") is False


@patch("src.fetch.yf.Ticker")
@patch("src.fetch._get_connection")
def test_fetch_price_network_call(mock_get_conn, mock_ticker_cls, mock_yf_data, mock_db):
    """Test that data is fetched from API when cache is disabled or empty."""
    # Setup mock DB connection
    mock_get_conn.return_value = mock_db
    
    # Setup mock yfinance
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_yf_data
    mock_ticker_cls.return_value = mock_ticker
    
    # 1. Fetch with cache=False
    df = fetch_price_data("AAPL", use_cache=False)
    assert len(df) == 10
    mock_ticker.history.assert_called_once()
    
    # Verify nothing written to DB
    cursor = mock_db.execute("SELECT count(*) FROM price_cache")
    assert cursor.fetchone()[0] == 0


@patch("src.fetch.yf.Ticker")
@patch("src.fetch._get_connection")
def test_fetch_price_caches_data(mock_get_conn, mock_ticker_cls, mock_yf_data, mock_db):
    """Test that API data is written to cache."""
    # Prevent the code from closing our in-memory DB by wrapping it
    mock_conn = MagicMock(wraps=mock_db)
    mock_conn.close = MagicMock()
    mock_get_conn.return_value = mock_conn
    
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_yf_data
    mock_ticker_cls.return_value = mock_ticker
    
    fetch_price_data("AAPL", period="1mo", use_cache=True)
    
    # Check data in DB
    cursor = mock_db.execute("SELECT count(*) FROM price_cache WHERE ticker='AAPL'")
    assert cursor.fetchone()[0] == 10
    
    cursor = mock_db.execute("SELECT period FROM cache_metadata WHERE ticker='AAPL'")
    assert cursor.fetchone()[0] == "1mo"


@patch("src.fetch.yf.Ticker")
@patch("src.fetch._get_connection")
def test_fetch_reads_from_cache(mock_get_conn, mock_ticker_cls, mock_yf_data, mock_db):
    """Test that data is read from cache on subsequent calls."""
    # Prevent code from closing DB
    mock_conn = MagicMock(wraps=mock_db)
    mock_conn.close = MagicMock()
    mock_get_conn.return_value = mock_conn
    
    # Pre-populate cache
    now = datetime.now()
    now_str = now.isoformat()
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    
    mock_db.execute(
        "INSERT INTO cache_metadata (ticker, last_fetched, period) VALUES (?, ?, ?)",
        ("AAPL", now_str, "1y")
    )
    # Insert one fake row
    mock_db.execute(
        "INSERT INTO price_cache (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("AAPL", yesterday, 100, 110, 90, 105, 5000)
    )
    mock_db.commit()
    
    # Call fetch
    df = fetch_price_data("AAPL", period="1y", use_cache=True)
    
    # Should NOT call yfinance
    mock_ticker_cls.assert_not_called()
    assert len(df) == 1
    assert df.iloc[0]["Close"] == 105.0


def test_fetch_invalid_ticker():
    """Test error handling for empty ticker."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        fetch_price_data("")
