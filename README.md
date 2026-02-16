# Financial Analytics Dashboard

A professional-grade Python application for ingesting, analyzing, and visualizing equity market data. Built with **Streamlit**, **Plotly**, and **Pandas**, featuring persistent local caching and robust risk/return metrics.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Financial+Analytics+Dashboard+Preview)

## Features

- **Interactive Dashboard**: 
  - Real-time switching between tickers (AAPL, MSFT, SPY, etc.) and time periods.
  - Multi-panel visualizations: Price + SMA, Cumulative Returns, Rolling Volatility, Drawdown.
- **Robust Data Pipeline**:
  - Integration with **Yahoo Finance API**.
  - **SQLite Caching Layer**: Minimizes API calls and enables offline analysis for cached data.
  - Automatic cache invalidation based on time (6h expiry) and requested period coverage.
- **Financial Analysis**:
  - **Risk Metrics**: Annualized Volatility, Max Drawdown, Sharpe Ratio.
  - **Technical Indicators**: Simple Moving Averages (20, 50, 200-day).

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KyleWareEng/Market-Analytics-Platform.git
   cd market-analytics-platform
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the dashboard application:

```bash
streamlit run src/dashboard.py
```

The application will open in your default browser at `http://localhost:8501`.

## Development

To run the test suite:
## Testing

To run the unit tests, first install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Then run pytest:

```bash
python -m pytest tests/ -v
```

## Project Structure

```
.
├── data/               # Local SQLite database (gitignored)
├── src/
│   ├── analysis.py     # Financial computations and logic
│   ├── dashboard.py    # Streamlit UI and Plotly visualizations
│   └── fetch.py        # Data ingestion and caching layer
├── tests/              # Unit tests
├── requirements.txt    # Production dependencies
└── README.md           # Project documentation
```

## License

[MIT](LICENSE)
