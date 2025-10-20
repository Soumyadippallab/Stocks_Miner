import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ripser import ripser
from persim import bottleneck
import yfinance as yf
import os

def takens_embedding(data, dim, time_delay):
    """Takens embedding for time series data."""
    n = len(data) - (dim - 1) * time_delay
    if n <= 0:
        raise ValueError("Not enough data points for given dimension and time delay.")
    return np.array([data[i:i + dim * time_delay:time_delay] for i in range(n)])

def process_stock_data(ticker, start_date="2010-01-01", end_date="2020-12-31",
                       window_size=50, embedding_dim=3, time_delay=1):
    """
    Perform TDA crash detection on stock price data using Takens embedding, ripser, 
    and bottleneck distances to detect potential crashes.
    
    Parameters:
    - ticker: str, stock symbol (e.g., 'TSLA')
    - start_date, end_date: str, date range
    - window_size: int, sliding window size for TDA
    - embedding_dim: int, embedding dimension for Takens embedding
    - time_delay: int, time delay for embedding
    """

    # Temporary CSV path
    csv_file = f"{ticker.replace('.', '_')}.csv"

    # Delete existing CSV if exists
    if os.path.exists(csv_file):
        os.remove(csv_file)

    # Download data
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(str(ticker), start=start_date, end=end_date)[["Close"]]
    df.to_csv(csv_file)
    print(f"Temporary file {csv_file} saved.")

    print(f"Data length: {len(df)} from {df.index[0]} to {df.index[-1]}...")

    # Normalize close prices
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    close_prices_scaled = scaler.fit_transform(close_prices)

    # Takens embedding
    embedded_data = takens_embedding(close_prices_scaled, embedding_dim, time_delay)

    # Sliding window + persistence diagrams
    persistence_diagrams = []
    bottleneck_distances = []

    for i in range(len(embedded_data) - window_size + 1):
        window_data = embedded_data[i:i + window_size].reshape(window_size, embedding_dim)
        result = ripser(window_data, maxdim=2, thresh=0.2)
        dgms = result['dgms']
        persistence_diagrams.append(dgms)

        if i > 0:
            distance = bottleneck(persistence_diagrams[i-1][1], dgms[1])  # 1st homology
            bottleneck_distances.append(distance)

    bottleneck_distances = np.array(bottleneck_distances)

    # Dynamic crash threshold
    crash_threshold = np.mean(bottleneck_distances) + 2 * np.std(bottleneck_distances)

    # Detect crashes
    crash_indices = np.where(bottleneck_distances > crash_threshold)[0] + 1

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # Stock price plot
    ax1.plot(df.index, df['Close'], label="Close Price")
    ax1.set_title(f"{ticker} Stock Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.grid(True)

    # Bottleneck distances plot
    ax2.plot(range(1, len(bottleneck_distances) + 1), bottleneck_distances, color='orange', label='Bottleneck Distance')
    ax2.axhline(crash_threshold, color='red', linestyle='--', label='Crash Threshold')
    ax2.scatter(crash_indices, bottleneck_distances[crash_indices - 1], color='red', label='Detected Crashes')
    ax2.set_title("Bottleneck Distance for Crash Detection")
    ax2.set_xlabel("Window Index")
    ax2.set_ylabel("Bottleneck Distance")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Delete temporary CSV
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print(f"Temporary file {csv_file} deleted.")

    print("TDA crash detection complete.")
























'''# tda_crash_detection.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import SlidingWindow
from sklearn.preprocessing import MinMaxScaler

def process_stock_data(data, start_date=None, end_date=None, save_csv=False):
    """
    Perform TDA crash detection on stock price data.
    
    Parameters:
    - data: pd.Series or pd.DataFrame (stock closing prices)
    - start_date, end_date: optional, for info/logging
    - save_csv: bool, whether to save temporary CSV
    """

    # Ensure data is a Series
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]  # take first column
    
    print(f"Fetching data for Ticker\n{data.head()}\nData length: {len(data)} from {data.index[0]} to {data.index[-1]}...")

    # Optional: save to CSV temporarily
    temp_file = "temp_stock_data.csv"
    if save_csv:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        data.to_csv(temp_file)
        print(f"Temporary file {temp_file} created.")

    # Scaling for TDA
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

    # Sliding window embedding
    window_size = min(30, len(data_scaled)//10)  # adaptive window size
    sw = SlidingWindow(size=window_size, stride=1)
    X = sw.fit_transform(data_scaled)

    # Compute Vietoris-Rips persistence
    print("Computing Vietoris-Rips persistence...")
    vr = VietorisRipsPersistence(homology_dimensions=[0,1])
    diagrams = vr.fit_transform(X)

    # Plot persistence diagram
    plt.figure(figsize=(8,6))
    for i, dim in enumerate([0,1]):
        plt.scatter(diagrams[0][diagrams[0][:,0]==dim][:,1], diagrams[0][diagrams[0][:,0]==dim][:,2], label=f"H{dim}")
    plt.title("Persistence Diagram")
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.legend()
    plt.show()

    # Simple crash detection: use significant 0D persistence as threshold
    h0 = diagrams[0][diagrams[0][:,0]==0]
    threshold = np.percentile(h0[:,2]-h0[:,1], 90)  # top 10% lifetimes
    significant_indices = np.where((h0[:,2]-h0[:,1]) >= threshold)[0]
    crash_points = [int(i) for i in significant_indices if i < len(data)]

    # Plot time series with crash points
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data.values, label="Price")
    if crash_points:
        plt.scatter(data.index[crash_points], data.values[crash_points], color='red', label="Potential Crashes")
    plt.title("Stock Price with Potential Crashes")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Delete temporary CSV if created
    if save_csv and os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Temporary file {temp_file} deleted.")

    print("TDA crash detection complete.")'''
