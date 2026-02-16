"""
Market data ingestion module.

Fetches equity price data from Yahoo Finance with SQLite caching
to minimize redundant API calls and improve pipeline reliability.
"""

import sqlite3
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "market_data.db"
CACHE_EXPIRY_HOURS = 6

# Map periods to approximate days for validity checking
# NOTE: Used for checking if our cached data covers the requested period.
# e.g. if we have '1y' cached, we can serve '1mo' requests from it.
PERIOD_DAYS = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "10y": 3650,
    "ytd": 365,  # Approximate
    "max": 99999,
}


def _get_connection() -> sqlite3.Connection:
    """Create or connect to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_cache (
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
        CREATE TABLE IF NOT EXISTS cache_metadata (
            ticker TEXT PRIMARY KEY,
            last_fetched TEXT,
            period TEXT
        )
    """)
    conn.commit()
    return conn


def _is_cache_valid(conn: sqlite3.Connection, ticker: str, requested_period: str) -> bool:
    """
    Check if cached data is fresh and covers the requested period.
    
    A cache hit requires:
    1. Data exists for the ticker.
    2. Data was fetched recently (within CACHE_EXPIRY_HOURS).
    3. The cached period is at least as long as the requested period (e.g., '1y' covers '3mo').
    """
    try:
        cursor = conn.execute(
            "SELECT last_fetched, period FROM cache_metadata WHERE ticker = ?",
            (ticker,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return False

        last_fetched_str, cached_period = row
        last_fetched = datetime.fromisoformat(last_fetched_str)
        
        # Check freshness
        if (datetime.now() - last_fetched) > timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Cache expired for {ticker} (Last fetched: {last_fetched})")
            return False

        # Check coverage
        cached_days = PERIOD_DAYS.get(cached_period, 0)
        requested_days = PERIOD_DAYS.get(requested_period, 0)
        
        if cached_days < requested_days:
            logger.info(f"Cache insufficient for {ticker}: have {cached_period}, need {requested_period}")
            return False
            
        return True
        
    except sqlite3.Error as e:
        logger.warning(f"Error checking cache validity: {e}")
        return False


def _read_from_cache(conn: sqlite3.Connection, ticker: str, period: str) -> pd.DataFrame:
    """Read price data from SQLite cache, filtering by requested period."""
    # Determine start date based on period
    days = PERIOD_DAYS.get(period, 365)
    start_date = (datetime.now() - timedelta(days=days)).isoformat()
    
    query = """
        SELECT date, open, high, low, close, volume 
        FROM price_cache 
        WHERE ticker = ? AND date >= ? 
        ORDER BY date ASC
    """
    
    df = pd.read_sql_query(
        query,
        conn,
        params=(ticker, start_date),
        parse_dates=["date"],
        index_col="date",
    )
    
    # Capitalize columns to match yfinance output
    df.columns = [c.capitalize() for c in df.columns]
    return df


def _write_to_cache(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame, period: str) -> None:
    """Write price data to SQLite cache."""
    records = []
    now = datetime.now().isoformat()
    
    for date, row in df.iterrows():
        records.append((
            ticker,
            date.isoformat(),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            int(row["Volume"]),
            now,
            period
        ))
    
    # Use transaction for atomicity
    try:
        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO price_cache 
                (ticker, date, open, high, low, close, volume, fetched_at, period_fetched) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.execute(
                "INSERT OR REPLACE INTO cache_metadata (ticker, last_fetched, period) VALUES (?, ?, ?)",
                (ticker, now, period),
            )
    except sqlite3.Error as e:
        logger.error(f"Failed to write to cache for {ticker}: {e}")
        raise


def fetch_price_data(
    ticker: str,
    period: str = "1y",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given equity ticker.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g., 'AAPL', 'MSFT').
    period : str
        Lookback period. Accepts yfinance period strings: '1mo', '3mo', '6mo', '1y', '2y', '5y'.
    use_cache : bool
        If True, attempts to read from local SQLite cache before hitting the API.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].

    Raises
    ------
    ValueError
        If ticker is empty or data retrieval returns no results.
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError(f"Invalid ticker: {ticker!r}")

    ticker = ticker.upper().strip()
    
    if use_cache:
        try:
            conn = _get_connection()
            if _is_cache_valid(conn, ticker, period):
                logger.info(f"Cache hit for {ticker} (period={period})")
                df = _read_from_cache(conn, ticker, period)
                if not df.empty:
                    conn.close()
                    return df
            else:
                logger.info(f"Cache miss for {ticker}")
            conn.close()
        except sqlite3.Error as e:
            logger.warning(f"Cache access failed: {e}. Falling back to API.")

    logger.info(f"Fetching from API for {ticker} (period={period})")
    
    try:
        stock = yf.Ticker(ticker)
        # auto_adjust=True accounts for splits and dividends
        df = stock.history(period=period, auto_adjust=True)
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {ticker} from Yahoo Finance: {e}") from e

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. This may be an invalid symbol or network issue.")

    # Validate output schema
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
         # Sometimes yfinance returns lower case columns or different structures
        logger.warning(f"Missing columns {missing_cols} for {ticker}. Attempting to normalize.")
        
    df = df[required_cols].copy()
    df.index.name = "date"
    # Ensure timezone-naive datetime index for consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Write to cache
    if use_cache:
        try:
            conn = _get_connection()
            _write_to_cache(conn, ticker, df, period)
            conn.close()
            logger.info(f"Cached {len(df)} rows for {ticker}")
        except Exception as e:
            logger.error(f"Failed to cache data for {ticker}: {e}")

    return df


def fetch_multiple(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """
    Fetch data for multiple tickers.
    
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping ticker symbol to price DataFrame.
    """
    results = {}
    for t in tickers:
        try:
            results[t.upper()] = fetch_price_data(t, period=period)
        except ValueError as e:
            logger.error(f"Skipping {t}: {e}")
    return results