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
    # Sort results by absolute CAGR (to include negatives by magnitude)
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    # Print top performers
    print("\nTop performers by CAGR:")
    for ticker, cagr in results[:top_x]:
        print(f"{ticker}: {cagr:.2f}%")
    
    # Attempt to plot - wrap in try-except to handle rendering errors
    # Plot top performers with simple, soothing style
    top_tickers = [x[0] for x in results[:top_x]]
    top_cagr = [x[1] for x in results[:top_x]]
    
    # Check if data range is too small and skip plotting to avoid rendering errors
    if top_cagr:
        min_cagr = min(top_cagr)
        max_cagr = max(top_cagr)
        data_range = max_cagr - min_cagr
        
        # If range is extremely small, skip plotting
        if data_range < 1.0:  # Less than 1% range
            print("\n⚠️  Plot not shown: CAGR values are too similar for meaningful visualization.")
            print("    The CAGR values above are still accurate. Try a longer date range for better visualization.")
            return
    
    try:
        # Elegant, eye-catching color - single color for simplicity
        elegant_purple = '#8B5CF6' # Rich, sophisticated purple
        fig, ax = plt.subplots(figsize=(10, 8))  # Increased height for better label fit in Colab
      
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
        # Rotate x labels (reduced angle for better fit)
        num_bars = len(top_tickers)
        ax.set_xticks(range(num_bars))
        ax.set_xticklabels(top_tickers, rotation=30, ha='right', fontsize=10)  # Reduced to 30°
        
        # Set minimum range to prevent microscopic scales
        MIN_RANGE = 10  # Minimum 10% range
        
        if data_range < MIN_RANGE:
            # Center the data and use minimum range
            center = (max_cagr + min_cagr) / 2
            ax.set_ylim(center - MIN_RANGE/2, center + MIN_RANGE/2)
        else:
            # Use padding for larger ranges
            padding = 0.15 * data_range
            ax.set_ylim(min_cagr - padding, max_cagr + padding)
        
        # Simple value labels with adjusted vertical alignment for negatives
        for bar, cagr_val in zip(bars, top_cagr):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, height,
                    f"{cagr_val:.1f}%",
                    ha='center', va=va,
                    fontsize=9, color='#333')
        # Remove unnecessary spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Manual bottom margin adjustment for rotated labels
        plt.subplots_adjust(bottom=0.18)
        
        # Suppress tight_layout warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Tight layout.*')
            plt.tight_layout()
        
        # Try to display the plot with error handling
        try:
            from .utils import smart_show
            smart_show()
        except Exception as display_error:
            # Catch errors that occur during display/rendering
            error_msg = str(display_error)
            if "Image size" in error_msg or "too large" in error_msg:
                print("\n⚠️  Plot not shown due to Colab/Notebook rendering limitations.")
                print("    The CAGR values above are still accurate. Try a longer date range for visualization.")
            else:
                print(f"\n⚠️  Plot display failed: {error_msg}")
        finally:
            plt.close('all')
    except Exception as e:
        # If plotting construction fails
        plt.close('all')
        error_msg = str(e)
        print(f"\n⚠️  Plot creation failed: {error_msg}")

# import pandas as pd
# import yfinance as yf
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import sys
# from .utils import get_default_dates, calculate_cagr
# def analyze_nse_stocks(num_tickers=10, top_x=5, start_date=None, end_date=None):
#     # Get default dates if not provided
#     start_date, end_date = get_default_dates() if not start_date or not end_date else (start_date, end_date)
#     print("Fetching NSE stock list...")
#     try:
#         url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
#         import urllib.request
#         import io
#         req = urllib.request.Request(
#             url,
#             headers={
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#             }
#         )
#         with urllib.request.urlopen(req, timeout=30) as response:
#             nse_list = pd.read_csv(io.BytesIO(response.read()))
#         tickers = nse_list["SYMBOL"].head(num_tickers).tolist()
#     except Exception as e:
#         print(f"Error fetching NSE stock list: {e}")
#         return
#     print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...\n")
#     results = []
#     for ticker in tqdm(tickers, desc="Downloading stock data", unit="ticker"):
#         ticker_yf = ticker + ".NS"
#         try:
#             full_data = yf.download(ticker_yf, start=start_date, end=end_date, progress=False)
#             data = full_data["Close"].squeeze().dropna()
#             if len(data) < 2:
#                 print(f"{ticker} has insufficient data for CAGR calculation, skipping.")
#                 continue
#             start_price = data.iloc[0]
#             end_price = data.iloc[-1]
#             start_dt = data.index[0]
#             end_dt = data.index[-1]
#             actual_days = (end_dt - start_dt).days
#             # Calculate CAGR
#             cagr = calculate_cagr(start_price, end_price, actual_days)
#             results.append((ticker, cagr))
#         except Exception as e:
#             print(f"Error fetching {ticker}: {e}")
#     if not results:
#         print("No valid stock data found for CAGR calculation.")
#         return
#     # Sort results by absolute CAGR (to include negatives by magnitude)
#     results.sort(key=lambda x: abs(x[1]), reverse=True)
#     # Print top performers
#     print("\nTop performers by CAGR:")
#     for ticker, cagr in results[:top_x]:
#         print(f"{ticker}: {cagr:.2f}%")
#     # Plot top performers with simple, soothing style
#     top_tickers = [x[0] for x in results[:top_x]]
#     top_cagr = [x[1] for x in results[:top_x]]
#     # Elegant, eye-catching color - single color for simplicity
#     elegant_purple = '#8B5CF6' # Rich, sophisticated purple
#     fig, ax = plt.subplots(figsize=(10, 8))  # Increased height for better label fit in Colab
  
#     # Create clean bars
#     bars = ax.bar(top_tickers, top_cagr,
#                    color=elegant_purple,
#                    alpha=0.85,
#                    edgecolor='none')
#     # Minimal labels
#     ax.set_xlabel("Ticker", fontsize=11)
#     ax.set_ylabel("CAGR (%)", fontsize=11)
#     ax.set_title(f"Top {top_x} Stocks by CAGR", fontsize=13, pad=15)
#     # Very light grid
#     ax.grid(True, alpha=0.2, linestyle='--', axis='y')
#     ax.set_axisbelow(True)
#     # Rotate x labels (reduced angle for better fit)
#     num_bars = len(top_tickers)
#     ax.set_xticks(range(num_bars))
#     ax.set_xticklabels(top_tickers, rotation=30, ha='right', fontsize=10)  # Reduced to 30°
#     # Adjust y limits to include negatives
#     if top_cagr:
#         min_cagr = min(top_cagr)
#         max_cagr = max(top_cagr)
#         padding = 0.15 * max(abs(min_cagr), abs(max_cagr))
#         ax.set_ylim(min_cagr - padding, max_cagr + padding)
#     else:
#         ax.set_ylim(-100, 100)
#     # Simple value labels with adjusted vertical alignment for negatives
#     for bar, cagr_val in zip(bars, top_cagr):
#         height = bar.get_height()
#         va = 'bottom' if height >= 0 else 'top'
#         ax.text(bar.get_x() + bar.get_width()/2, height,
#                 f"{cagr_val:.1f}%",
#                 ha='center', va=va,
#                 fontsize=9, color='#333')
#     # Remove unnecessary spines
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     # Manual bottom margin adjustment for rotated labels
#     plt.subplots_adjust(bottom=0.18)
#     plt.tight_layout()  # Now applies without overflow
#     #plt.show(block=True)
#     #plt.close('all')
#     #sys.exit(0)
#     from .utils import smart_show
#     smart_show()
#     plt.close('all')


# import pandas as pd
# import yfinance as yf
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import sys
# from .utils import get_default_dates, calculate_cagr
# def analyze_nse_stocks(num_tickers=10, top_x=5, start_date=None, end_date=None):
#     # Get default dates if not provided
#     start_date, end_date = get_default_dates() if not start_date or not end_date else (start_date, end_date)
#     print("Fetching NSE stock list...")
#     try:
#         url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
#         import urllib.request
#         import io
#         req = urllib.request.Request(
#             url,
#             headers={
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#             }
#         )
#         with urllib.request.urlopen(req, timeout=30) as response:
#             nse_list = pd.read_csv(io.BytesIO(response.read()))
#         tickers = nse_list["SYMBOL"].head(num_tickers).tolist()
#     except Exception as e:
#         print(f"Error fetching NSE stock list: {e}")
#         return
#     print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...\n")
#     results = []
#     for ticker in tqdm(tickers, desc="Downloading stock data", unit="ticker"):
#         ticker_yf = ticker + ".NS"
#         try:
#             full_data = yf.download(ticker_yf, start=start_date, end=end_date, progress=False)
#             data = full_data["Close"].squeeze().dropna()
#             if len(data) < 2:
#                 print(f"{ticker} has insufficient data for CAGR calculation, skipping.")
#                 continue
#             start_price = data.iloc[0]
#             end_price = data.iloc[-1]
#             start_dt = data.index[0]
#             end_dt = data.index[-1]
#             actual_days = (end_dt - start_dt).days
#             # Calculate CAGR
#             cagr = calculate_cagr(start_price, end_price, actual_days)
#             results.append((ticker, cagr))
#         except Exception as e:
#             print(f"Error fetching {ticker}: {e}")
#     if not results:
#         print("No valid stock data found for CAGR calculation.")
#         return
#     # Sort results by absolute CAGR (to include negatives by magnitude)
#     results.sort(key=lambda x: abs(x[1]), reverse=True)
#     # Print top performers
#     print("\nTop performers by CAGR:")
#     for ticker, cagr in results[:top_x]:
#         print(f"{ticker}: {cagr:.2f}%")
#     # Plot top performers with simple, soothing style
#     top_tickers = [x[0] for x in results[:top_x]]
#     top_cagr = [x[1] for x in results[:top_x]]
#     # Elegant, eye-catching color - single color for simplicity
#     elegant_purple = '#8B5CF6' # Rich, sophisticated purple
#     fig, ax = plt.subplots(figsize=(10, 6))
  
#     # Create clean bars
#     bars = ax.bar(top_tickers, top_cagr,
#                    color=elegant_purple,
#                    alpha=0.85,
#                    edgecolor='none')
#     # Minimal labels
#     ax.set_xlabel("Ticker", fontsize=11)
#     ax.set_ylabel("CAGR (%)", fontsize=11)
#     ax.set_title(f"Top {top_x} Stocks by CAGR", fontsize=13, pad=15)
#     # Very light grid
#     ax.grid(True, alpha=0.2, linestyle='--', axis='y')
#     ax.set_axisbelow(True)
#     # Rotate x labels
#     num_bars = len(top_tickers)
#     ax.set_xticks(range(num_bars))
#     ax.set_xticklabels(top_tickers, rotation=45, ha='right', fontsize=10)
#     # Adjust y limits to include negatives
#     if top_cagr:
#         min_cagr = min(top_cagr)
#         max_cagr = max(top_cagr)
#         padding = 0.15 * max(abs(min_cagr), abs(max_cagr))
#         ax.set_ylim(min_cagr - padding, max_cagr + padding)
#     else:
#         ax.set_ylim(-100, 100)
#     # Simple value labels with adjusted vertical alignment for negatives
#     for bar, cagr_val in zip(bars, top_cagr):
#         height = bar.get_height()
#         va = 'bottom' if height >= 0 else 'top'
#         ax.text(bar.get_x() + bar.get_width()/2, height,
#                 f"{cagr_val:.1f}%",
#                 ha='center', va=va,
#                 fontsize=9, color='#333')
#     # Remove unnecessary spines
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.tight_layout()
#     #plt.show(block=True)
#     #plt.close('all')
#     #sys.exit(0)
#     from .utils import smart_show
#     smart_show()
#     plt.close('all')


# import pandas as pd
# import yfinance as yf
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import sys
# from .utils import get_default_dates, calculate_cagr

# def analyze_nse_stocks(num_tickers=10, top_x=5, start_date=None, end_date=None):
#     # Get default dates if not provided
#     start_date, end_date = get_default_dates() if not start_date or not end_date else (start_date, end_date)

#     print("Fetching NSE stock list...")
#     try:
#         url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
#         import urllib.request
#         import io
#         req = urllib.request.Request(
#             url, 
#             headers={
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#             }
#         )
#         with urllib.request.urlopen(req, timeout=30) as response:
#             nse_list = pd.read_csv(io.BytesIO(response.read()))
#         tickers = nse_list["SYMBOL"].head(num_tickers).tolist()
#     except Exception as e:
#         print(f"Error fetching NSE stock list: {e}")
#         return

#     print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...\n")
#     results = []

#     for ticker in tqdm(tickers, desc="Downloading stock data", unit="ticker"):
#         ticker_yf = ticker + ".NS"
#         try:
#             full_data = yf.download(ticker_yf, start=start_date, end=end_date, progress=False)
#             data = full_data["Close"].squeeze().dropna()
#             if len(data) < 2:
#                 print(f"{ticker} has insufficient data for CAGR calculation, skipping.")
#                 continue

#             start_price = data.iloc[0]
#             end_price = data.iloc[-1]
#             start_dt = data.index[0]
#             end_dt = data.index[-1]
#             actual_days = (end_dt - start_dt).days

#             # Calculate CAGR
#             cagr = calculate_cagr(start_price, end_price, actual_days)
#             results.append((ticker, cagr))

#         except Exception as e:
#             print(f"Error fetching {ticker}: {e}")

#     if not results:
#         print("No valid stock data found for CAGR calculation.")
#         return

#     # Sort results by CAGR
#     results.sort(key=lambda x: x[1], reverse=True)

#     # Print top performers
#     print("\nTop performers by CAGR:")
#     for ticker, cagr in results[:top_x]:
#         print(f"{ticker}: {cagr:.2f}%")

#     # Plot top performers with simple, soothing style
#     top_tickers = [x[0] for x in results[:top_x]]
#     top_cagr = [x[1] for x in results[:top_x]]
#     # Elegant, eye-catching color - single color for simplicity
#     elegant_purple = '#8B5CF6' # Rich, sophisticated purple
#     fig, ax = plt.subplots(figsize=(10, 6))
   
#     # Create clean bars
#     bars = ax.bar(top_tickers, top_cagr,
#                    color=elegant_purple,
#                    alpha=0.85,
#                    edgecolor='none')
#     # Minimal labels
#     ax.set_xlabel("Ticker", fontsize=11)
#     ax.set_ylabel("CAGR (%)", fontsize=11)
#     ax.set_title(f"Top {top_x} Stocks by CAGR", fontsize=13, pad=15)
#     # Very light grid
#     ax.grid(True, alpha=0.2, linestyle='--', axis='y')
#     ax.set_axisbelow(True)
#     # Rotate x labels
#     num_bars = len(top_tickers)
#     ax.set_xticks(range(num_bars))
#     ax.set_xticklabels(top_tickers, rotation=45, ha='right', fontsize=10)
#     # Adjust y limits
#     max_cagr = max(top_cagr) if top_cagr else 100
#     ax.set_ylim(0, max_cagr * 1.15)
#     # Simple value labels
#     for bar, cagr_val in zip(bars, top_cagr):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2, height,
#                 f"{cagr_val:.1f}%",
#                 ha='center', va='bottom',
#                 fontsize=9, color='#333')
#     # Remove unnecessary spines
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.tight_layout()
#     #plt.show(block=True)
#     #plt.close('all')
#     #sys.exit(0)
#     from .utils import smart_show
#     smart_show()
#     plt.close('all')
