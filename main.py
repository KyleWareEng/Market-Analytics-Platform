"""
Financial Analytics Dashboard — Main entry point.

Usage:
    python main.py                    # Run Streamlit dashboard
    python main.py --ticker AAPL      # Quick CLI analysis for a ticker
    python main.py --export AAPL      # Export charts as HTML
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_cli(ticker: str, period: str, export: bool = False) -> None:
    """Run a quick CLI analysis and optionally export charts."""
    from src.fetch import fetch_price_data
    from src.analysis import compute_summary_stats
    from src.dashboard import (
        build_dashboard,
        build_price_chart,
        build_volatility_chart,
        build_drawdown_chart,
    )

    logger.info(f"Analysing {ticker} ({period})")

    df = fetch_price_data(ticker, period=period)
    prices = df["Close"]
    stats = compute_summary_stats(prices)

    print(f"\n{'='*50}")
    print(f"  {ticker} — Summary Statistics ({period})")
    print(f"{'='*50}")
    for key, val in stats.items():
        if isinstance(val, float):
            if "return" in key or "volatility" in key or "drawdown" in key or "day" in key:
                print(f"  {key:<25s} {val:>10.2%}")
            else:
                print(f"  {key:<25s} {val:>10.4f}")
        else:
            print(f"  {key:<25s} {val:>10}")
    print(f"{'='*50}\n")

    if export:
        fig = build_dashboard(prices, ticker)
        out_path = f"data/{ticker}_dashboard.html"
        fig.write_html(out_path)
        logger.info(f"Dashboard exported to {out_path}")
        print(f"✓ Exported interactive dashboard → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Financial Analytics Dashboard")
    parser.add_argument("--ticker", type=str, help="Run CLI analysis for a ticker")
    parser.add_argument("--period", type=str, default="1y", help="Lookback period (default: 1y)")
    parser.add_argument("--export", action="store_true", help="Export dashboard as HTML")
    args = parser.parse_args()

    if args.ticker:
        run_cli(args.ticker, args.period, export=args.export)
    else:
        # Launch Streamlit
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/dashboard.py",
            "--server.headless", "true",
        ])


if __name__ == "__main__":
    main()
