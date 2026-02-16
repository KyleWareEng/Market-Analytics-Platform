"""
Time-series analysis and risk metrics module.

Provides financial computations including moving averages,
volatility, drawdown analysis, and return metrics.
"""

import pandas as pd
import numpy as np


def compute_returns(prices: pd.Series) -> pd.Series:
    """
    Compute simple daily returns from a price series.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices.

    Returns
    -------
    pd.Series
        Daily percentage returns.
    """
    if prices.empty:
        raise ValueError("Price series is empty.")
    return prices.pct_change().dropna()


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Compute cumulative returns from a daily return series.

    Parameters
    ----------
    returns : pd.Series
        Daily percentage returns.

    Returns
    -------
    pd.Series
        Cumulative return series (starting from 0).
    """
    return (1 + returns).cumprod() - 1


def compute_moving_averages(
    prices: pd.Series,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute simple moving averages for specified window sizes.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices.
    windows : list[int], optional
        Rolling window sizes in trading days. Defaults to [20, 50, 200].

    Returns
    -------
    pd.DataFrame
        DataFrame with a column for each moving average.
    """
    if windows is None:
        windows = [20, 50, 200]

    result = pd.DataFrame(index=prices.index)
    for w in windows:
        if w < 1:
            raise ValueError(f"Window size must be positive, got {w}")
        result[f"SMA_{w}"] = prices.rolling(window=w).mean()
    return result


def compute_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """
    Compute rolling historical volatility.

    Parameters
    ----------
    returns : pd.Series
        Daily percentage returns.
    window : int
        Rolling window in trading days.
    annualize : bool
        If True, scale to annualized volatility.
    trading_days : int
        Number of trading days per year for annualization.

    Returns
    -------
    pd.Series
        Rolling volatility series.
    """
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(trading_days)
    return vol


def compute_drawdown(prices: pd.Series) -> pd.DataFrame:
    """
    Compute drawdown series and identify maximum drawdown.

    Drawdown measures the peak-to-trough decline of an investment,
    expressed as a percentage from the running maximum.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'wealth_index': normalized cumulative value (starting at 1)
        - 'running_max': running peak of wealth index
        - 'drawdown': percentage drawdown from peak (negative values)
    """
    if prices.empty:
        raise ValueError("Price series is empty.")

    wealth_index = prices / prices.iloc[0]
    running_max = wealth_index.cummax()
    drawdown = (wealth_index - running_max) / running_max

    return pd.DataFrame({
        "wealth_index": wealth_index,
        "running_max": running_max,
        "drawdown": drawdown,
    })


def compute_max_drawdown(prices: pd.Series) -> float:
    """
    Compute the maximum drawdown as a single scalar value.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices.

    Returns
    -------
    float
        Maximum drawdown (negative value, e.g., -0.25 for 25% drawdown).
    """
    dd = compute_drawdown(prices)
    return float(dd["drawdown"].min())


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily percentage returns.
    risk_free_rate : float
        Annualized risk-free rate (default 0).
    trading_days : int
        Trading days per year.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    if returns.std() == 0:
        return 0.0
    
    # Adjust risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    # Simple approximation if risk_free_rate is provided as annual percentage (e.g. 0.05)
    # Often for Sharpe, people just subtract daily_rf = annualized_rf / 252.
    # We'll use the arithmetic mean approach for simplicity and consistency with common practice
    daily_rf_simple = risk_free_rate / trading_days
    
    excess = returns - daily_rf_simple
    
    # Scale both numerator (mean) and denominator (std) to annual
    # Annualized Sharpe = sqrt(252) * (mean_daily_excess / std_daily_excess)
    # derived from: (mean * 252) / (std * sqrt(252))
    
    return float(np.sqrt(trading_days) * (excess.mean() / excess.std()))

def compute_summary_stats(prices: pd.Series) -> dict:
    """
    Compute a comprehensive summary of risk and return metrics.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices.

    Returns
    -------
    dict
        Dictionary containing:
        - total_return: cumulative percentage return
        - annualized_return: CAGR
        - annualized_volatility: annualized standard deviation
        - sharpe_ratio: risk-adjusted return metric
        - max_drawdown: worst peak-to-trough decline
        - best_day: highest single-day return
        - worst_day: lowest single-day return
    """
    returns = compute_returns(prices)
    n_days = len(returns)
    trading_days = 252

    total_return = float((prices.iloc[-1] / prices.iloc[0]) - 1)
    years = n_days / trading_days
    
    # CAGR calculation handling short periods
    if years > 0:
        annualized_return = float((1 + total_return) ** (1 / max(years, 1e-6)) - 1)
    else:
        annualized_return = 0.0
        
    annualized_vol = float(returns.std() * np.sqrt(trading_days))

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": compute_sharpe_ratio(returns),
        "max_drawdown": compute_max_drawdown(prices),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "trading_days": n_days,
    }
