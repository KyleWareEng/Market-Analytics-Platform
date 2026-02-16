# Market Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modular financial analytics dashboard for equity market analysis. Features interactive visualisations, intelligent caching, and comprehensive risk metrics.

![Dashboard Preview](images/preview_spy_1y_page1.png)

---

## Features

**Interactive Dashboard**
- Real-time ticker switching (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, SPY, QQQ)
- Configurable lookback periods (3 months to max history)
- Dark-themed financial UI inspired by professional trading platforms

**Financial Metrics**
- Total & Annualised Returns (CAGR)
- Rolling Volatility (20-day annualised)
- Sharpe Ratio, Maximum Drawdown
- Moving Averages (SMA 20, 50, 200)

**Three Operating Modes**
- **Streamlit Dashboard** — Interactive web UI
- **CLI Analysis** — Terminal-based summary statistics
- **HTML Export** — Standalone shareable files

**Intelligent Caching**
- SQLite-based persistent cache with 6-hour expiry
- Smart coverage detection (cached 1y data serves 3mo requests)
- Offline capability for previously fetched data

---

## Technical Architecture

<p align="center">
  <img src="images/architecture_diagram.png" alt="Technical Architecture" width="500">
</p>

**Design Principles:** Separation of concerns · Cache-first architecture · Full type hints · Comprehensive testing · Structured logging

---

## Installation

```bash
git clone https://github.com/KyleWareEng/market-analytics-platform.git
cd market-analytics-platform

python -m venv env
source env/bin/activate  # Windows: .\env\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Interactive Dashboard

```bash
python main.py
# Opens browser at http://localhost:8501
```

### CLI Analysis

```bash
python main.py --ticker TSLA --period 2y
```

```
==================================================
  TSLA — Summary Statistics (2y)
==================================================
  total_return                      45.23%
  annualized_return                 20.15%
  annualized_volatility             45.67%
  sharpe_ratio                      0.4412
  max_drawdown                     -35.12%
  best_day                           8.45%
  worst_day                         -9.12%
  trading_days                         504
==================================================
```

### HTML Export

```bash
python main.py --ticker AAPL --period 1y --export
# Creates: data/AAPL_dashboard.html
```

### Programmatic Use

```python
from src.fetch import fetch_price_data
from src.analysis import compute_summary_stats
from src.dashboard import build_dashboard

# Fetch data
df = fetch_price_data("MSFT", period="1y")
prices = df["Close"]

# Compute metrics
stats = compute_summary_stats(prices)
print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")

# Generate chart
fig = build_dashboard(prices, "MSFT")
fig.show()
```

---

## Project Structure

```
market-analytics-platform/
├── src/
│   ├── fetch.py          # Data ingestion & SQLite caching
│   ├── analysis.py       # Financial computations
│   └── dashboard.py      # Plotly visualisations & Streamlit UI
├── tests/
│   ├── test_analysis.py  # Analysis unit tests
│   └── test_fetch.py     # Data fetching tests
├── images/               # Dashboard screenshots
├── main.py               # Application entry point
├── requirements.txt
└── requirements-dev.txt
```

---

## Technical Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **Data** | yfinance, SQLite, Pandas, NumPy |
| **Visualisation** | Plotly, Streamlit |
| **Testing** | pytest, coverage |
| **Code Quality** | black, flake8, mypy |

---

## Financial Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Annualised Volatility** | σ_daily × √252 | Risk measure |
| **Sharpe Ratio** | √252 × (mean / std) | Return per unit risk (>1 good) |
| **Maximum Drawdown** | (wealth - peak) / peak | Worst decline from peak |

---

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

Test coverage includes: returns computation, moving averages, volatility scaling, drawdown constraints, Sharpe ratio edge cases, and summary statistics validation.

---

## Dashboard Views

| Main Dashboard | Individual Charts |
|----------------|-------------------|
| ![SPY](images/preview_spy_1y_page1.png) | ![GOOGL](images/preview_googl_1y_page1.png) |

| Cumulative Return | Drawdown Analysis |
|-------------------|-------------------|
| ![Returns](images/preview_cumulativereturn_spy_1y_page3.png) | ![Drawdown](images/preview_drawdown_spy_1y_page5.png) |

---

## AI-Augmented Workflow

This project was developed using AI-assisted tooling (Claude, Antigravity) to accelerate iteration, refactor code, and validate architectural decisions. All implementations were reviewed, tested, and verified to ensure correctness and maintain ownership of system design.

---

## What I Learned

- Designing modular systems with clear separation of concerns
- Implementing caching strategies and database schemas
- Financial mathematics: annualisation, risk-adjusted returns, drawdown analysis
- Production Python: type hints, logging, testing, code quality tooling
- Building interactive dashboards with Streamlit and Plotly
- Effectively leveraging AI coding assistants in development workflows

---

## Future Enhancements

- [ ] Sortino Ratio, Value at Risk (VaR), Beta
- [ ] Portfolio analysis with correlation matrix
- [ ] Docker containerisation
- [ ] Cloud deployment

---

## Contact

**Kyle Ware**  
[LinkedIn](https://linkedin.com/in/kyleaware) · [Email](mailto:kyle.ware@outlook.com) · [GitHub](https://github.com/KyleWareEng)

MEng Automotive Engineering | Data Science & Quantitative Finance
