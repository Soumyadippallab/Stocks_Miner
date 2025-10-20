# Stocks Miner

Stocks_miner is a Python-based tool for financial data analysis, leveraging libraries like yfinance, pandas, and ripser to process stock market data. It provides command-line interfaces (CLIs) for four main functionalities:

1. Analyze SENSEX and NIFTY indices with statistical metrics and visualizations.  
2. Analyze NSE stocks with company-wise and sector-wise metrics.  
3. Analyze randomly selected stocks or sectors.  
4. Detect market crashes using Topological Data Analysis (TDA) on user-provided stock data.



## Installation

You can install the package locally by navigating to the Stock_Miner2 directory and running:

pip install .

Or using the traditional setup:

python setup.py install



## Usage

1. Analyze Market Indices  
stocks-miner-indices --start_date 2024-01-01 --end_date 2025-09-21

2. Analyze NSE Stocks  
stocks-miner-nse --num_tickers 10 --top_x 5 --start_date 2024-01-01 --end_date 2025-09-21

3. Analyze Random Stocks/Sectors  
stocks-miner-random --k 5 --selection_type companies --start_date 2024-01-01 --end_date 2025-09-21

4. TDA Crash Detection  
stocks-miner-tda "data/stock_data.csv" --start_date 2024-01-01 --end_date 2025-09-21



## Directory Structure

Stock_Miner2/  
├── data/                   (User-provided CSV/XLSX files for TDA)  
├── examples/               (Example scripts or notebooks, optional)  
├── stocks_miner/           (Main Python package)  
│   ├── __init__.py  
│   ├── cli.py  
│   ├── market_indices.py  
│   ├── nse_stocks.py  
│   ├── random_stocks.py  
│   ├── tda_crash_detection.py  
│   └── utils.py  
├── tests/                  (Unit tests, optional)  
├── LICENSE  
├── README.md  
├── requirements.txt  
└── setup.py  



## Dependencies

The package requires the following Python libraries: pandas, numpy, yfinance, scikit-learn, scipy, matplotlib, seaborn, yahooquery, tqdm, requests, ripser, persim.

Install them using:

pip install -r requirements.txt



## Features

- Market Indices Analysis: Compute moving averages, volatility, trend analysis, linear regression for trends, t-tests, and visualize SENSEX and NIFTY trends.  
- NSE Stocks Analysis: Company-wise and sector-wise metrics including returns, moving averages, volatility, Sharpe ratio, CAGR, and correlation matrices. Outputs top performers by CAGR and Sharpe ratio.  
- Random Stocks/Sectors Analysis: Random selection of companies or sectors with evaluation metrics and visualizations of returns vs. volatility and Sharpe ratios.  
- TDA Crash Detection: Detect potential market crashes using Takens embedding, persistence diagrams, Wasserstein and Bottleneck distances, and sliding window embeddings. Visualizations of price and distance metrics with crash indicators.



## Author

Soumyadip Das



## License

This project is licensed under the MIT License. See the LICENSE file for details.
