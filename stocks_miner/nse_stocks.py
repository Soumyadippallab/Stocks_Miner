import pandas as pd
import yfinance as yf
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from .utils import get_default_dates, calculate_cagr

def analyze_nse_stocks(num_tickers=10, top_x=5, start_date=None, end_date=None):
    # Get default dates if not provided
    start_date, end_date = get_default_dates() if not start_date or not end_date else (start_date, end_date)

    print("Fetching NSE stock list...")
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        import urllib.request
        import io
        req = urllib.request.Request(
            url, 
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            nse_list = pd.read_csv(io.BytesIO(response.read()))
        tickers = nse_list["SYMBOL"].head(num_tickers).tolist()
    except Exception as e:
        print(f"Error fetching NSE stock list: {e}")
        return

    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...\n")
    results = []

    for ticker in tqdm(tickers, desc="Downloading stock data", unit="ticker"):
        ticker_yf = ticker + ".NS"
        try:
            full_data = yf.download(ticker_yf, start=start_date, end=end_date, progress=False)
            data = full_data["Close"].squeeze().dropna()
            if len(data) < 2:
                print(f"{ticker} has insufficient data for CAGR calculation, skipping.")
                continue

            start_price = data.iloc[0]
            end_price = data.iloc[-1]
            start_dt = data.index[0]
            end_dt = data.index[-1]
            actual_days = (end_dt - start_dt).days

            # Calculate CAGR
            cagr = calculate_cagr(start_price, end_price, actual_days)
            results.append((ticker, cagr))

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    if not results:
        print("No valid stock data found for CAGR calculation.")
        return

    # Sort results by CAGR
    results.sort(key=lambda x: x[1], reverse=True)

    # Print top performers
    print("\nTop performers by CAGR:")
    for ticker, cagr in results[:top_x]:
        print(f"{ticker}: {cagr:.2f}%")

    # Plot top performers with simple, soothing style
    top_tickers = [x[0] for x in results[:top_x]]
    top_cagr = [x[1] for x in results[:top_x]]
    # Elegant, eye-catching color - single color for simplicity
    elegant_purple = '#8B5CF6' # Rich, sophisticated purple
    fig, ax = plt.subplots(figsize=(10, 6))
   
    # Create clean bars
    bars = ax.bar(top_tickers, top_cagr,
                   color=elegant_purple,
                   alpha=0.85,
                   edgecolor='none')
    # Minimal labels
    ax.set_xlabel("Ticker", fontsize=11)
    ax.set_ylabel("CAGR (%)", fontsize=11)
    ax.set_title(f"Top {top_x} Stocks by CAGR", fontsize=13, pad=15)
    # Very light grid
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    # Rotate x labels
    num_bars = len(top_tickers)
    ax.set_xticks(range(num_bars))
    ax.set_xticklabels(top_tickers, rotation=45, ha='right', fontsize=10)
    # Adjust y limits
    max_cagr = max(top_cagr) if top_cagr else 100
    ax.set_ylim(0, max_cagr * 1.15)
    # Simple value labels
    for bar, cagr_val in zip(bars, top_cagr):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f"{cagr_val:.1f}%",
                ha='center', va='bottom',
                fontsize=9, color='#333')
    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show(block=True)
    plt.close('all')
    sys.exit(0)