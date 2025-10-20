import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import calculate_cagr
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def analyze_market_indices(start_date=None, end_date=None):
    indices = {"SENSEX": "^BSESN", "NIFTY": "^NSEI"}
    df_indices = {}

    print("Fetching market indices data...")
    for name, ticker in indices.items():
        print(f"Fetching {name}...")
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)["Close"]
        if not data.empty:
            df_indices[name] = data
        else:
            print(f"Warning: No data found for {name}, skipping...")

    # Only continue if we have at least one valid index
    if not df_indices:
        print("No valid index data available. Exiting...")
        return

    # Combine into DataFrame
    df = pd.concat(df_indices.values(), axis=1)
    df.columns = list(df_indices.keys())

    # Calculate daily returns
    df_returns = df.pct_change().dropna()

    # Calculate CAGR
    print("\nCAGR for indices:")
    for name in df.columns:
        cagr = calculate_cagr(df[name].iloc[0], df[name].iloc[-1], len(df))
        print(f"{name}: {cagr:.2f}%")

    # Plot price trends
    plt.figure(figsize=(12, 6))
    for name in df.columns:
        plt.plot(df[name], label=name)
    plt.title("Market Indices Price Trends")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot daily returns
    plt.figure(figsize=(12, 6))
    for name in df_returns.columns:
        plt.plot(df_returns[name], label=name)
    plt.title("Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_returns.corr(), annot=True, cmap="coolwarm")
    plt.title("Return Correlation Between Indices")
    plt.tight_layout()
    plt.show()
