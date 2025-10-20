import random
import pandas as pd
import yfinance as yf
import urllib.request
import io
from .utils import get_default_dates, calculate_cagr
from datetime import datetime

def analyze_random_stocks_or_sectors(k=5, selection_type="companies", start_date=None, end_date=None):
    """
    Fast random stock CAGR analysis with progress prints.
    Handles immediate termination on Ctrl+C.
    """
    # Set default dates
    start_date, end_date = get_default_dates() if not start_date or not end_date else (start_date, end_date)

    # Fetch NSE list
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        req = urllib.request.Request(
            url, 
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            nse_list = pd.read_csv(io.BytesIO(response.read()))
    except Exception as e:
        print(f"Error fetching NSE list: {e}")
        return

    # Select random tickers
    if selection_type.lower() == "companies":
        choices = random.sample(nse_list["SYMBOL"].tolist(), k)  # Note: Column is 'SYMBOL', not 'Symbol'
    else:
        if "Industry" in nse_list.columns:
            choices = random.sample(nse_list["Industry"].dropna().unique().tolist(), k)
        else:
            print("No 'Industry' column found; falling back to random companies.")
            choices = random.sample(nse_list["SYMBOL"].tolist(), k)

    print(f"Selected {selection_type}: {choices}\n")

    results = []

    try:
        for idx, item in enumerate(choices, start=1):
            ticker_yf = item + ".NS"
            print(f"[{idx}/{k}] Fetching {ticker_yf}... ", end="")
            
            # Fetch weekly data (faster)
            full_data = yf.download(ticker_yf, start=start_date, end=end_date, interval="1wk", progress=False)
            data = full_data["Close"].squeeze().dropna()
            if len(data) < 2:
                print("No data, skipping.")
                continue

            start_price = data.iloc[0]
            end_price = data.iloc[-1]
            start_dt = data.index[0]
            end_dt = data.index[-1]
            actual_days = (end_dt - start_dt).days

            # Calculate CAGR
            cagr = calculate_cagr(start_price, end_price, actual_days)
            results.append((item, cagr))
            print(f"CAGR = {cagr:.2f}%")

    except KeyboardInterrupt:
        print("\nProcess terminated by user (Ctrl+C).")
        return

    if not results:
        print("\nNo valid stock data found.")
        return

    # Sort and show top performers
    results.sort(key=lambda x: x[1], reverse=True)
    print("\nTop performers by CAGR:")
    for item, cagr in results:
        print(f"{item}: {cagr:.2f}%")