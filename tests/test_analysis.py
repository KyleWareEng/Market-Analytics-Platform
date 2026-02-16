"""
Unit tests for analysis module.

Run: python -m pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np

from src.analysis import (
    compute_returns,
    compute_cumulative_returns,
    compute_moving_averages,
    compute_volatility,
    compute_drawdown,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_summary_stats,
)


@pytest.fixture
def sample_prices():
    """Generate a simple deterministic price series for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 100)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="Close")


@pytest.fixture
def monotonic_up():
    """Price series that only goes up."""
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    return pd.Series(np.linspace(100, 200, 50), index=dates)


@pytest.fixture
def monotonic_down():
    """Price series that only goes down."""
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    return pd.Series(np.linspace(200, 100, 50), index=dates)


class TestReturns:
    def test_returns_length(self, sample_prices):
        ret = compute_returns(sample_prices)
        assert len(ret) == len(sample_prices) - 1

    def test_returns_no_nan(self, sample_prices):
        ret = compute_returns(sample_prices)
        assert not ret.isna().any()

    def test_empty_series_raises(self):
        with pytest.raises(ValueError):
            compute_returns(pd.Series(dtype=float))

    def test_cumulative_returns_start_near_zero(self, sample_prices):
        ret = compute_returns(sample_prices)
        cum = compute_cumulative_returns(ret)
        assert abs(cum.iloc[0]) < 0.1  # first value is just the first return


class TestMovingAverages:
    def test_default_windows(self, sample_prices):
        sma = compute_moving_averages(sample_prices)
        assert "SMA_20" in sma.columns
        assert "SMA_50" in sma.columns
        assert "SMA_200" in sma.columns

    def test_custom_windows(self, sample_prices):
        sma = compute_moving_averages(sample_prices, windows=[5, 10])
        assert list(sma.columns) == ["SMA_5", "SMA_10"]

    def test_sma_values(self, sample_prices):
        sma = compute_moving_averages(sample_prices, windows=[5])
        expected = sample_prices.rolling(5).mean()
        pd.testing.assert_series_equal(sma["SMA_5"], expected, check_names=False)

    def test_invalid_window_raises(self, sample_prices):
        with pytest.raises(ValueError):
            compute_moving_averages(sample_prices, windows=[0])


class TestVolatility:
    def test_volatility_shape(self, sample_prices):
        ret = compute_returns(sample_prices)
        vol = compute_volatility(ret, window=20)
        assert len(vol) == len(ret)

    def test_annualized_greater_than_daily(self, sample_prices):
        ret = compute_returns(sample_prices)
        vol_ann = compute_volatility(ret, window=20, annualize=True)
        vol_daily = compute_volatility(ret, window=20, annualize=False)
        # Annualized should be larger on average
        assert vol_ann.dropna().mean() > vol_daily.dropna().mean()


class TestDrawdown:
    def test_drawdown_columns(self, sample_prices):
        dd = compute_drawdown(sample_prices)
        assert set(dd.columns) == {"wealth_index", "running_max", "drawdown"}

    def test_drawdown_never_positive(self, sample_prices):
        dd = compute_drawdown(sample_prices)
        assert (dd["drawdown"] <= 0).all()

    def test_monotonic_up_zero_drawdown(self, monotonic_up):
        dd = compute_drawdown(monotonic_up)
        assert np.allclose(dd["drawdown"].values, 0.0)

    def test_monotonic_down_drawdown(self, monotonic_down):
        max_dd = compute_max_drawdown(monotonic_down)
        expected = (100 - 200) / 200  # -0.5
        assert abs(max_dd - expected) < 0.01

    def test_max_drawdown_scalar(self, sample_prices):
        max_dd = compute_max_drawdown(sample_prices)
        assert isinstance(max_dd, float)
        assert max_dd <= 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_drawdown(pd.Series(dtype=float))


class TestSharpeRatio:
    def test_positive_returns_positive_sharpe(self, monotonic_up):
        ret = compute_returns(monotonic_up)
        sr = compute_sharpe_ratio(ret)
        assert sr > 0

    def test_negative_returns_negative_sharpe(self, monotonic_down):
        ret = compute_returns(monotonic_down)
        sr = compute_sharpe_ratio(ret)
        assert sr < 0

    def test_zero_vol_returns_zero(self):
        ret = pd.Series([0.0] * 50)
        assert compute_sharpe_ratio(ret) == 0.0


class TestSummaryStats:
    def test_keys(self, sample_prices):
        stats = compute_summary_stats(sample_prices)
        expected_keys = {
            "total_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "max_drawdown", "best_day", "worst_day", "trading_days",
        }
        assert set(stats.keys()) == expected_keys

    def test_types(self, sample_prices):
        stats = compute_summary_stats(sample_prices)
        for k, v in stats.items():
            assert isinstance(v, (int, float)), f"{k} has unexpected type {type(v)}"
