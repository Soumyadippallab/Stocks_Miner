import argparse
from .market_indices import analyze_market_indices
from .nse_stocks import analyze_nse_stocks
from .random_stocks import analyze_random_stocks_or_sectors
import os
import pandas as pd
import yfinance as yf

def main():
    parser = argparse.ArgumentParser(description="Stocks Miner CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Indices
    parser_indices = subparsers.add_parser("indices")
    parser_indices.add_argument("--start_date")
    parser_indices.add_argument("--end_date")

    # NSE
    parser_nse = subparsers.add_parser("nse")
    parser_nse.add_argument("--num_tickers", type=int, default=10)
    parser_nse.add_argument("--top_x", type=int, default=5)
    parser_nse.add_argument("--start_date")
    parser_nse.add_argument("--end_date")

    # Random
    parser_random = subparsers.add_parser("random")
    parser_random.add_argument("--k", type=int, default=5)
    parser_random.add_argument("--selection_type", default="companies")
    parser_random.add_argument("--start_date")
    parser_random.add_argument("--end_date")

    # TDA
    parser_tda = subparsers.add_parser("tda")
    parser_tda.add_argument("--ticker", required=True, help="Stock ticker symbol (e.g., DABUR.NS)")
    parser_tda.add_argument("--start_date")
    parser_tda.add_argument("--end_date")

    args = parser.parse_args()

    if args.command == "indices":
        analyze_market_indices(args.start_date, args.end_date)
    elif args.command == "nse":
        analyze_nse_stocks(args.num_tickers, args.top_x, args.start_date, args.end_date)
    elif args.command == "random":
        analyze_random_stocks_or_sectors(args.k, args.selection_type, args.start_date, args.end_date)
    elif args.command == "tda":
        from .tda_crash_detection import process_stock_data  # Ensure this function accepts Series or DataFrame

        # Default dates if not provided
        start_date = args.start_date if args.start_date else "2024-01-01"
        end_date = args.end_date if args.end_date else "2025-09-21"

        # CSV filename for the ticker
        temp_csv = f"{args.ticker.replace('.', '_')}.csv"
        downloaded_temp = False

        # Check if CSV exists
        if os.path.exists(temp_csv):
            print(f"Found existing data for {args.ticker}, using it...")
            data = pd.read_csv(temp_csv, index_col=0, parse_dates=True)
        else:
            print(f"Downloading data for {args.ticker} from {start_date} to {end_date}...")
            ticker_str = str(args.ticker)  # ensure it's a string
            data = yf.download(ticker_str, start=start_date, end=end_date)["Close"]
            data.to_csv(temp_csv)
            print(f"Temporary file {temp_csv} saved.")
            downloaded_temp = True

        # Call TDA crash detection
        process_stock_data(args.ticker, start_date, end_date)

        # Delete temporary CSV if it was freshly downloaded
        if downloaded_temp:
            try:
                os.remove(temp_csv)
                print(f"Temporary file {temp_csv} deleted.")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_csv}: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
