"""
Interactive dashboard module.

Builds Plotly visualizations for price, volatility, and drawdown analysis.
Serves as a Streamlit application.
"""

import logging
import sys
from pathlib import Path

# -- Path setup ------------------------------------------------------------
# Add project root to sys.path to allow importing from src when running as script
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

# -- Logging setup ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from src.analysis import (
    compute_cumulative_returns,
    compute_moving_averages,
    compute_volatility,
    compute_drawdown,
    compute_summary_stats,
)
from src.fetch import fetch_price_data

COLORS = {
    "bg": "#0d1117",
    "paper": "#161b22",
    "grid": "#21262d",
    "text": "#c9d1d9",
    "muted": "#8b949e",
    "price": "#58a6ff",
    "sma20": "#f0883e",
    "sma50": "#bc8cff",
    "sma200": "#3fb950",
    "vol": "#f0883e",
    "dd": "#f85149",
    "dd_fill": "rgba(248,81,73,0.15)",
    "cumret": "#58a6ff",
    "accent": "#58a6ff",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    plot_bgcolor=COLORS["bg"],
    paper_bgcolor=COLORS["paper"],
    font=dict(family="JetBrains Mono, Fira Code, monospace", color=COLORS["text"], size=12),
    margin=dict(l=60, r=30, t=50, b=40),
    xaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    yaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    hovermode="x unified",
)


def _apply_layout(fig: go.Figure, title: str, yaxis_title: str = "") -> go.Figure:
    """Apply consistent layout styling to a figure."""
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=16), x=0.02, xanchor="left"),
        yaxis_title=yaxis_title,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=COLORS["muted"]),
            orientation="h",
            y=1.02,  # Move legend above graph to prevent overlap
            x=1,
            xanchor="right",
        ),
    )
    return fig


# -- Individual chart builders ---------------------------------------------

def build_price_chart(prices: pd.Series, ticker: str, windows: list[int] | None = None) -> go.Figure:
    """Build an interactive price chart with moving average overlays."""
    if windows is None:
        windows = [20, 50, 200]

    sma = compute_moving_averages(prices, windows)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        name="Close", line=dict(color=COLORS["price"], width=2),
    ))

    sma_colors = [COLORS["sma20"], COLORS["sma50"], COLORS["sma200"]]
    for i, w in enumerate(windows):
        col = f"SMA_{w}"
        if col in sma.columns:
            # Filter out NaNs for cleaner plotting
            valid_sma = sma[col].dropna()
            fig.add_trace(go.Scatter(
                x=valid_sma.index, y=valid_sma.values,
                name=col, line=dict(color=sma_colors[i % len(sma_colors)], width=1.2, dash="dot"),
            ))

    return _apply_layout(fig, f"{ticker} - Price & Moving Averages", yaxis_title="Price ($)")


def build_volatility_chart(prices: pd.Series, ticker: str, window: int = 20) -> go.Figure:
    """Build a rolling annualized volatility chart."""
    returns = prices.pct_change().dropna()
    vol = compute_volatility(returns, window=window)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vol.index, y=vol.values,
        name=f"{window}d Vol (ann.)",
        line=dict(color=COLORS["vol"], width=1.8),
        fill="tozeroy",
        fillcolor="rgba(240,136,62,0.1)",
    ))

    fig.update_yaxes(tickformat=".0%")
    return _apply_layout(fig, f"{ticker} - Rolling {window}d Annualized Volatility", yaxis_title="Volatility")


def build_drawdown_chart(prices: pd.Series, ticker: str) -> go.Figure:
    """Build a drawdown chart showing peak-to-trough declines."""
    dd = compute_drawdown(prices)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd["drawdown"].values,
        name="Drawdown",
        line=dict(color=COLORS["dd"], width=1.5),
        fill="tozeroy",
        fillcolor=COLORS["dd_fill"],
    ))

    # Add annotation for max drawdown
    max_dd_idx = dd["drawdown"].idxmin()
    max_dd_val = dd["drawdown"].min()
    
    if pd.notna(max_dd_val):
        fig.add_annotation(
            x=max_dd_idx, y=max_dd_val,
            text=f"Max DD: {max_dd_val:.1%}",
            showarrow=True, arrowhead=2,
            font=dict(color=COLORS["dd"], size=11, weight="bold"),
            arrowcolor=COLORS["dd"],
            yshift=-10, # Shift down slightly to avoid covering line
        )

    fig.update_yaxes(tickformat=".0%")
    return _apply_layout(fig, f"{ticker} - Drawdown from Peak", yaxis_title="Drawdown")


def build_cumulative_return_chart(prices: pd.Series, ticker: str) -> go.Figure:
    """Build cumulative return chart."""
    returns = prices.pct_change().dropna()
    cum_ret = compute_cumulative_returns(returns)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_ret.index, y=cum_ret.values,
        name="Cumulative Return",
        line=dict(color=COLORS["cumret"], width=2),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
    ))

    fig.update_yaxes(tickformat=".0%")
    return _apply_layout(fig, f"{ticker} - Cumulative Return", yaxis_title="Return")


# -- Combined dashboard figure --------------------------------------------

def build_dashboard(prices: pd.Series, ticker: str) -> go.Figure:
    """
    Build a combined 4-panel dashboard figure.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices.
    ticker : str
        Ticker symbol.

    Returns
    -------
    go.Figure
        Combined subplot figure.
    """
    returns = prices.pct_change().dropna()
    cum_ret = compute_cumulative_returns(returns)
    vol = compute_volatility(returns, window=20)
    dd = compute_drawdown(prices)
    sma = compute_moving_averages(prices, [20, 50, 200])

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08, # Increased spacing
        row_heights=[0.35, 0.20, 0.20, 0.25],
        subplot_titles=["Price & Moving Averages", "Cumulative Return", "Rolling Volatility (20d)", "Drawdown"],
    )

    # Row 1 — Price + SMAs
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        name="Close", line=dict(color=COLORS["price"], width=2),
    ), row=1, col=1)

    for i, (col, c) in enumerate(zip(
        ["SMA_20", "SMA_50", "SMA_200"],
        [COLORS["sma20"], COLORS["sma50"], COLORS["sma200"]],
    )):
        if col in sma.columns:
            valid_sma = sma[col].dropna()
            fig.add_trace(go.Scatter(
                x=valid_sma.index, y=valid_sma.values,
                name=col, line=dict(color=c, width=1, dash="dot"),
            ), row=1, col=1)

    # Row 2 — Cumulative return
    fig.add_trace(go.Scatter(
        x=cum_ret.index, y=cum_ret.values,
        name="Cum Return", line=dict(color=COLORS["cumret"], width=1.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
    ), row=2, col=1)

    # Row 3 — Volatility
    fig.add_trace(go.Scatter(
        x=vol.index, y=vol.values,
        name="Volatility", line=dict(color=COLORS["vol"], width=1.5),
        fill="tozeroy", fillcolor="rgba(240,136,62,0.08)",
    ), row=3, col=1)

    # Row 4 — Drawdown
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd["drawdown"].values,
        name="Drawdown", line=dict(color=COLORS["dd"], width=1.5),
        fill="tozeroy", fillcolor=COLORS["dd_fill"],
    ), row=4, col=1)

    # Add Max DD annotation to Row 4
    max_dd_val = dd["drawdown"].min()
    max_dd_idx = dd["drawdown"].idxmin()
    
    if pd.notna(max_dd_val):
        fig.add_annotation(
            x=max_dd_idx, y=max_dd_val,
            text=f"Max DD: {max_dd_val:.1%}",
            xref="x4", yref="y4",
            showarrow=True, arrowhead=2,
            font=dict(color=COLORS["dd"], size=10, weight="bold"),
            arrowcolor=COLORS["dd"],
            yshift=-10,
        )

    # Format axes
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    fig.update_yaxes(tickformat=".0%", row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["paper"],
        font=dict(family="JetBrains Mono, Fira Code, monospace", color=COLORS["text"], size=11),
        height=1000,
        margin=dict(l=60, r=30, t=60, b=30),
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=COLORS["muted"]),
            orientation="h",
            y=-0.05,
        ),
        title=dict(
            text=f"{ticker} - Financial Analytics Dashboard",
            font=dict(size=18),
            x=0.02, xanchor="left",
        ),
    )

    # Style subplots grid
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=COLORS["grid"], showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor=COLORS["grid"], showgrid=True, row=i, col=1)

    return fig


# -- Streamlit app ---------------------------------------------------------

def run_streamlit_app():
    """Launch the Streamlit dashboard application."""
    st.set_page_config(
        page_title="Financial Analytics Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for clean, professional look
    st.markdown("""
        <style>
        .stApp { background-color: #0d1117; }
        .stMeasure { font-family: 'Source Sans Pro', sans-serif !important; }
        header { visibility: hidden; }
        </style>
    """, unsafe_allow_html=True)

    st.title("Financial Analytics Dashboard")
    st.markdown("Python-based pipeline for ingesting, analysing, and visualising equity market data.")

    # Sidebar for controls
    with st.sidebar:
        st.header("Configuration")
        ticker = st.selectbox(
            "Select Ticker",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "SPY", "QQQ"],
            index=7, # Default to SPY
        )
        period = st.selectbox(
            "Lookback Period", 
            ["3mo", "6mo", "1y", "2y", "5y", "10y", "max"], 
            index=2 # Default to 1y
        )
        
        st.markdown("---")
        st.markdown(
            "**Data Source**:\n"
            "Yahoo Finance API with local persistent SQLite caching.\n\n"
            "**Stack**:\n"
            "Pandas, Plotly, Streamlit"
        )

    # Main data fetch with error handling
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            df = fetch_price_data(ticker, period=period)
            logger.info(f"Loaded {len(df)} rows for {ticker}")
        except ValueError as e:
            st.error(f"Error fetching data: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            logger.exception("Unexpected error in dashboard execution")
            st.stop()

    prices = df["Close"]
    
    # Compute stats
    try:
        stats = compute_summary_stats(prices)
    except Exception as e:
        st.error(f"Error computing statistics: {e}")
        st.stop()

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{stats['total_return']:.1%}")
    c2.metric("Ann. Volatility", f"{stats['annualized_volatility']:.1%}")
    c3.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
    c4.metric("Max Drawdown", f"{stats['max_drawdown']:.1%}")

    # Main dashboard chart
    st.plotly_chart(build_dashboard(prices, ticker), width="stretch")

    # Detailed view expansion
    with st.expander("Explore Individual Charts"):
        t1, t2, t3, t4 = st.tabs(["Price", "Cumulative Return", "Volatility", "Drawdown"])
        with t1:
            st.plotly_chart(build_price_chart(prices, ticker), use_container_width=True)
        with t2:
            st.plotly_chart(build_cumulative_return_chart(prices, ticker), use_container_width=True)
        with t3:
            st.plotly_chart(build_volatility_chart(prices, ticker), use_container_width=True)
        with t4:
            st.plotly_chart(build_drawdown_chart(prices, ticker), use_container_width=True)

    # Raw data view
    with st.expander("View Raw Data"):
        st.dataframe(df.sort_index(ascending=False).head(100), use_container_width=True)


if __name__ == "__main__":
    run_streamlit_app()