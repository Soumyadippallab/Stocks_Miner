import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser
from persim import bottleneck, wasserstein
from .utils import validate_file_path

def takens_embedding(data, dimension, delay):
    """Perform Takens embedding on time series data."""
    n_points = len(data) - (dimension - 1) * delay
    if n_points <= 0:
        raise ValueError("Time series is too short for the given dimension and delay.")
    embedded = np.empty((n_points, dimension))
    for i in range(dimension):
        embedded[:, i] = data[i * delay: i * delay + n_points]
    return embedded

def create_sliding_windows(data, window_size=50):
    """Create sliding windows of points for TDA."""
    return np.array([data[i:i + window_size] for i in range(len(data) - window_size + 1)])

def euclidean_distance(a, b):
    """Compute Euclidean distance between two arrays."""
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    """Compute Manhattan distance between two arrays."""
    return np.sum(np.abs(a - b))

def compute_distances(items, distance_func):
    """Compute distances between consecutive items."""
    distances = []
    for i in range(1, len(items)):
        dist = distance_func(items[i-1], items[i])
        distances.append(dist)
    return np.array(distances)

def filter_non_finite_diagram(diagram):
    """Filter non-finite death times from persistence diagrams."""
    return diagram[np.isfinite(diagram[:, 1])]

def compute_price_changes(prices):
    """Compute daily percentage price changes."""
    return np.diff(prices) / prices[:-1] * 100

def process_stock_data(file_path=None, start_date=None, end_date=None, dimension=3, delay=1, window_size=50, homology_dims=[0, 1], bottleneck_threshold=0.01, other_threshold=0.05, price_drop_threshold=-7):
    """Detect market crashes using TDA and distance metrics."""
    # Prompt for file path if not provided
    if file_path is None:
        file_path = input("Enter path to your .xlsx or .csv file (e.g., C:/Users/YourName/Desktop/stock_data.csv): ").strip()
    
    # Validate file path
    file_path = validate_file_path(file_path)
    
    # Load data
    if file_path.lower().endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        data = pd.read_csv(file_path)
    
    # Validate columns
    if 'Close' not in data.columns or 'Date' not in data.columns:
        raise ValueError("Dataset must contain 'Date' and 'Close' columns.")
    
    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Filter data by date range if provided
    if start_date:
        data = data[data['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data['Date'] <= pd.to_datetime(end_date)]
    
    # Check if data is empty after filtering
    if data.empty:
        raise ValueError("No data available after applying date filters.")
    
    close_prices = pd.to_numeric(data['Close'], errors='coerce').fillna(method='ffill').values
    timestamps = data['Date'].values
    
    # Print data date range
    print(f"Data date range: {pd.Timestamp(timestamps.min()).strftime('%Y-%m-%d')} to {pd.Timestamp(timestamps.max()).strftime('%Y-%m-%d')}")
    
    # Normalize close prices
    close_prices = (close_prices - close_prices.mean()) / close_prices.std()
    
    # Compute daily percentage price changes
    price_changes = compute_price_changes(close_prices)
    
    # Apply Takens embedding
    embedded_points = takens_embedding(close_prices, dimension, delay)
    
    # Create sliding windows for TDA
    point_windows = create_sliding_windows(embedded_points, window_size)
    
    # Compute persistence diagrams
    persistence_diagrams = []
    for window in point_windows:
        result = ripser(window, maxdim=max(homology_dims))['dgms']
        diagram = np.vstack([result[dim] for dim in homology_dims if len(result[dim]) > 0])
        diagram = filter_non_finite_diagram(diagram)
        persistence_diagrams.append(diagram)
    
    # Compute distances
    euclidean_distances = compute_distances(embedded_points, euclidean_distance)
    manhattan_distances = compute_distances(embedded_points, manhattan_distance)
    bottleneck_distances = compute_distances(persistence_diagrams, bottleneck)
    wasserstein_distances = compute_distances(persistence_diagrams, wasserstein)
    
    # Set thresholds
    thresholds = {
        'wasserstein': other_threshold,
        'bottleneck': bottleneck_threshold,
        'euclidean': other_threshold,
        'manhattan': other_threshold
    }
    
    # Detect crash indices
    crash_indices = {
        'wasserstein': [],
        'bottleneck': [],
        'euclidean': [],
        'manhattan': []
    }
    
    # Euclidean/Manhattan: align price changes to distances
    for i, dist in enumerate(euclidean_distances):
        if dist > thresholds['euclidean'] and i < len(price_changes) and price_changes[i] < price_drop_threshold:
            crash_indices['euclidean'].append(i)
    for i, dist in enumerate(manhattan_distances):
        if dist > thresholds['manhattan'] and i < len(price_changes) and price_changes[i] < price_drop_threshold:
            crash_indices['manhattan'].append(i)
    
    # TDA: align price changes to windows
    for i, dist in enumerate(wasserstein_distances):
        price_change_idx = i + window_size - 1
        if dist > thresholds['wasserstein'] and price_change_idx < len(price_changes) and price_changes[price_change_idx] < price_drop_threshold:
            crash_indices['wasserstein'].append(i)
    for i, dist in enumerate(bottleneck_distances):
        price_change_idx = i + window_size - 1
        if dist > thresholds['bottleneck'] and price_change_idx < len(price_changes) and price_changes[price_change_idx] < price_drop_threshold:
            crash_indices['bottleneck'].append(i)
    
    # Prepare timestamps for plots
    dist_timestamps_euclidean = timestamps[(dimension - 1) * delay + 1:]
    dist_timestamps_tda = timestamps[(dimension - 1) * delay + window_size:]
    
    # Plotting
    stock_name = file_path.split('.')[0].split('/')[-1]
    plt.figure(figsize=(12, 24))
    
    # Plot stock price
    plt.subplot(5, 1, 1)
    price_timestamps = timestamps[(dimension - 1) * delay:]
    plt.plot(price_timestamps, close_prices[(dimension - 1) * delay:], label='Stock Price', color='blue')
    plt.xlabel("Date")
    plt.ylabel("Stock Price (Normalized)")
    plt.title(f"Stock Price ({stock_name})")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    
    # Helper function for distance plots
    def plot_distance(subplot_idx, distances, threshold, indices, label, color, dist_timestamps):
        plt.subplot(5, 1, subplot_idx)
        plt.plot(dist_timestamps[:len(distances)], distances, label=label, color=color, linewidth=2)
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Crash Threshold ({threshold})')
        plt.scatter([dist_timestamps[i] for i in indices], [distances[i] for i in indices], color='red', label='Detected Crashes', zorder=5)
        plt.fill_between(dist_timestamps[:len(distances)], 0, distances, where=distances > threshold, color='red', alpha=0.3, label='Crash Region')
        plt.title(f"{label} for Crash Detection ({stock_name})")
        plt.xlabel("Date")
        plt.ylabel("Distance")
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.xticks(rotation=45)
    
    # Plot distances
    plot_distance(2, wasserstein_distances, thresholds['wasserstein'], crash_indices['wasserstein'], 'Wasserstein Distance', 'orange', dist_timestamps_tda)
    plot_distance(3, bottleneck_distances, thresholds['bottleneck'], crash_indices['bottleneck'], 'Bottleneck Distance', 'cyan', dist_timestamps_tda)
    plot_distance(4, euclidean_distances, thresholds['euclidean'], crash_indices['euclidean'], 'Euclidean Distance', 'green', dist_timestamps_euclidean)
    plot_distance(5, manhattan_distances, thresholds['manhattan'], crash_indices['manhattan'], 'Manhattan Distance', 'purple', dist_timestamps_euclidean)
    
    plt.tight_layout()
    plt.show()
    
    # Print crash dates
    print("\n=== Crash Dates ===")
    for metric in ['wasserstein', 'bottleneck']:
        crash_ts = dist_timestamps_tda[crash_indices[metric]]
        formatted_dates = [pd.Timestamp(ts).strftime('%Y-%m-%d') for ts in crash_ts]
        print(f"{metric.capitalize()} Distance Crash Dates: {', '.join(formatted_dates) if formatted_dates else 'None'}")
    for metric in ['euclidean', 'manhattan']:
        crash_ts = dist_timestamps_euclidean[crash_indices[metric]]
        formatted_dates = [pd.Timestamp(ts).strftime('%Y-%m-%d') for ts in crash_ts]
        print(f"{metric.capitalize()} Distance Crash Dates: {', '.join(formatted_dates) if formatted_dates else 'None'}")
    
    return crash_indices, timestamps, close_prices