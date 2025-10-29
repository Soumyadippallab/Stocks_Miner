# Stocks Miner

**Stocks Miner** is a Python tool for financial data analysis. It uses libraries such as yfinance, pandas, matplotlib, seaborn, tqdm, scikit-learn, ripser, and persim to process stock market data. The package exposes a CLI with the following capabilities:

- Analyze market indices (SENSEX, NIFTY) with metrics and visualizations (CAGR, daily returns, correlations, price trends, heatmaps).
- Analyze NSE stocks with company-wise metrics and top-performer visualizations.
- Analyze randomly selected stocks or sectors and rank by CAGR.
- Detect market crashes using Topological Data Analysis (TDA) on stock time series via Takens' embedding, persistent homology, and bottleneck distances.

## Prerequisites

- Python 3.8+ is recommended (set this in `setup.py` with `python_requires` if you want to enforce it).

## Installation

Install directly from PyPI:

```bash
pip install stocks-miner
```

This will automatically install all dependencies. For local development from the repository root:

```bash
pip install .
```

**Note**: Run commands from the parent `Stocks_Miner/` directory (not inside `stocks_miner/`) using `python -m stocks_miner.cli <command>` to handle relative imports.

## Usage (examples)

### CLI Usage (VS Code/Terminal)

Analyze market indices (example dates):

```bash
python -m stocks_miner.cli indices --start_date 2024-01-01 --end_date 2025-09-21
```

Analyze NSE stocks:

```bash
python -m stocks_miner.cli nse --num_tickers 10 --top_x 5 --start_date 2024-01-01 --end_date 2025-09-21
```

Analyze random stocks/sectors:

```bash
python -m stocks_miner.cli random --k 5 --selection_type companies --start_date 2024-01-01 --end_date 2025-09-21
```

TDA crash detection for a ticker:

```bash
python -m stocks_miner.cli tda --ticker <TICKER> --start_date 2024-01-01 --end_date 2025-09-21
```

For full help and options:

```bash
python -m stocks_miner.cli --help
```

### Notebook Usage (Jupyter/Colab)

For notebooks, install the package (if not already installed). In a Colab cell:

```python
!pip install stocks-miner
```

Then, in subsequent cells, use the dotted imports (no path modifications needed after installation). Here's an example structure:

#### Cell 1: Setup and Import
```python
# Import the helper (notebook_helper.py is part of the stocks_miner package)
from stocks_miner.notebook_helper import setup_stocks_miner

# Setup Stocks Miner and get the modules
sm = setup_stocks_miner()

# Optional: Direct access to modules (already available via sm)
from stocks_miner import random_stocks
sm.random_stocks = random_stocks  # If needed for legacy compatibility

print("✓ All modules loaded!")
```

#### Cell 2: Analyze Market Indices
```python
# Analyze major market indices (NIFTY 50, SENSEX, etc.)
sm.market_indices.analyze_market_indices(
    start_date="2024-01-01", 
    end_date="2024-09-21"
)
```

#### Cell 3: Analyze NSE Stocks
```python
# Analyze top NSE stocks
sm.nse_stocks.analyze_nse_stocks(
    num_tickers=10,     # Number of stocks to analyze
    top_x=5,            # Top performers to identify
    start_date="2024-01-01", 
    end_date="2024-09-21"
)
```

#### Cell 4: Analyze Random Stocks or Sectors
```python
# Analyze random selection of stocks or sectors
sm.random_stocks.analyze_random_stocks_or_sectors(
    k=5,                          # Number to select
    selection_type='companies',   # 'companies' or 'sectors'
    start_date="2024-01-01", 
    end_date="2024-09-21"
)
```

#### Cell 5: TDA Crash Detection (Advanced)
```python
# Topological Data Analysis for crash detection
sm.tda_crash.process_stock_data(
    ticker='RELIANCE.NS',
    start_date='2020-01-01',
    end_date='2024-12-31',
    window_size=50,
    embedding_dim=3,
    time_delay=1
)
```

**Notes for Notebooks**:
- Use past dates for `end_date` (e.g., up to 2024-09-21) to ensure data availability.
- Restart the runtime after installation if dependency conflicts arise (common in Colab).
- For local Jupyter: Ensure the package is installed in your environment (`pip install .` from repo root).

## Directory structure

```
Stocks_Miner/
├── .ipynb_checkpoints/     # Jupyter notebook checkpoints
├── build/                  # Build artifacts
├── dist/                   # Distribution files
├── examples/               # Example scripts or notebooks
├── stocks_miner/           # Main Python package
│   ├── .ipynb_checkpoints/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── cli.py
│   ├── market_indices.py
│   ├── notebook_helper.py
│   ├── nse_stocks.py
│   ├── random_stocks.py
│   ├── tda_crash_detection.py
│   └── utils.py
├── stocks_miner.egg-info/  # Egg metadata
├── tests/                  # Unit tests (optional)
├── venv/                   # Virtual environment
├── .gitignore
├── code_sample.txt
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
└── stocks_miner.log
```

## Dependencies

Major dependencies include: `pandas`, `numpy`, `yfinance`, `matplotlib`, `seaborn`, `tqdm`, `scikit-learn`, `ripser`, `persim`, and `yahooquery`.

These are automatically installed via `pip install stocks-miner`.

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

## Author

Soumyadip Das, and <a href="https://github.com/cserajdeep"> Rajdeep Chatterjee </a>

## Organization

AmygdalaAI-India Lab [https://amygdalaaiindia.github.io/]

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.