# Stocks Miner

**Stocks Miner** is a Python-based tool for financial data analysis, leveraging libraries like yfinance, pandas, matplotlib, seaborn, tqdm, scikit-learn, ripser, and persim to process stock market data. It provides command-line interfaces (CLIs) for four main functionalities:

1. Analyze SENSEX and NIFTY indices with statistical metrics (CAGR, daily returns, correlations) and visualizations (price trends, returns plots, heatmaps).  
2. Analyze NSE stocks with company-wise metrics (CAGR) and visualizations of top performers.  
3. Analyze randomly selected stocks or sectors with CAGR rankings.  
4. Detect market crashes using Topological Data Analysis (TDA) on user-provided stock data via Takens embedding, persistent homology, and bottleneck distances.

## Installation

You can install the package locally by navigating to the Stocks_Miner directory and running:

```bash
pip install .
```

Or using the traditional setup:

```bash
python setup.py install
```

**Note**: Run commands from the parent `Stocks_Miner/` directory (not inside `stocks_miner/`) using `python -m stocks_miner.cli <command>` to handle relative imports.

## Usage

1. Analyze Market Indices  
   ```bash
   python -m stocks_miner.cli indices --start_date 2024-01-01 --end_date 2025-09-21
   ```

2. Analyze NSE Stocks  
   ```bash
   python -m stocks_miner.cli nse --num_tickers 10 --top_x 5 --start_date 2024-01-01 --end_date 2025-09-21
   ```

3. Analyze Random Stocks/Sectors  
   ```bash
   python -m stocks_miner.cli random --k 5 --selection_type companies --start_date 2024-01-01 --end_date 2025-09-21
   ```

# Stocks Miner

**Stocks Miner** is a Python tool for financial data analysis. It uses libraries such as yfinance, pandas, matplotlib, seaborn, tqdm, scikit-learn, ripser, and persim to process stock market data. The package exposes a CLI with the following capabilities:

- Analyze market indices (SENSEX, NIFTY) with metrics and visualizations (CAGR, daily returns, correlations, price trends, heatmaps).
- Analyze NSE stocks with company-wise metrics and top-performer visualizations.
- Analyze randomly selected stocks or sectors and rank by CAGR.
- Detect market crashes using Topological Data Analysis (TDA) on stock time series via Takens' embedding, persistent homology, and bottleneck distances.

## Prerequisites

- Python 3.8+ is recommended (set this in `setup.py` with `python_requires` if you want to enforce it).
- A virtual environment is strongly recommended to keep dependencies isolated.

## Installation

From the repository root (recommended):

```powershell
# Create and activate a virtual environment (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install in editable/development mode
python -m pip install -e .

# Or install dependencies from requirements.txt
python -m pip install -r requirements.txt
```

Notes:
- Prefer `python -m pip install -e .` or `python -m pip install .` over `python setup.py install` which is deprecated for most workflows.
- If you publish this package, pin dependency versions in `requirements.txt` or use `setup.cfg`/`pyproject.toml` to manage them.

## Usage (examples)

Analyze market indices (example dates):

```powershell
python -m stocks_miner.cli indices --start_date 2024-01-01 --end_date 2025-09-21
```

Analyze NSE stocks:

```powershell
python -m stocks_miner.cli nse --num_tickers 10 --top_x 5 --start_date 2024-01-01 --end_date 2025-09-21
```

Analyze random stocks/sectors:

```powershell
python -m stocks_miner.cli random --k 5 --selection_type companies --start_date 2024-01-01 --end_date 2025-09-21
```

TDA crash detection for a ticker:

```powershell
python -m stocks_miner.cli tda --ticker <TICKER> --start_date 2024-01-01 --end_date 2025-09-21
```

For full help and options:

```powershell
python -m stocks_miner.cli --help
```

## Directory structure

```
Stocks_Miner/
├── data/                   # User-provided CSV/XLSX files for TDA
├── examples/               # Example scripts or notebooks (optional)
├── stocks_miner/           # Main Python package
│   ├── __init__.py
│   ├── cli.py
│   ├── market_indices.py
│   ├── nse_stocks.py
│   ├── random_stocks.py
│   ├── tda_crash_detection.py
│   └── utils.py
├── tests/                  # Unit tests (optional)
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## Dependencies

Major dependencies include: `pandas`, `numpy`, `yfinance`, `matplotlib`, `seaborn`, `tqdm`, `scikit-learn`, `ripser`, `persim`, and `yahooquery`.

Install them via the `requirements.txt` file as shown above.

## Notes & recommendations

- Grammar/wording: "Takens' embedding" is a clearer form than "Takens embedding".
- Git: avoid committing generated artifacts such as virtual environments and Python bytecode. Add a `.gitignore` (example below) and remove tracked artifacts from the repo if present.

Example `.gitignore` snippet to add at the repo root:

```
# Virtual envs
venv/
.venv/

# Byte-compiled / caches
__pycache__/
*.py[cod]
*$py.class

# Packaging
dist/
build/
*.egg-info/
```

## Tests

If you have tests under `tests/`, run them with your test runner (e.g., `pytest`) after activating the virtual environment:

```powershell
pytest -q
```

## Author

Soumyadip Das

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
