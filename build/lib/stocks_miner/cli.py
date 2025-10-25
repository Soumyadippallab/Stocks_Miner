import argparse
from .market_indices import analyze_market_indices
from .nse_stocks import analyze_nse_stocks
from .random_stocks import analyze_random_stocks_or_sectors
from .tda_crash_detection import process_stock_data

def tda_command():
    parser = argparse.ArgumentParser(description="Perform TDA crash detection on stock data.")
    parser.add_argument('file_path', type=str, nargs='?', default=None, help="Path to .xlsx or .csv file")
    parser.add_argument('--start_date', type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end_date', type=str, default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    process_stock_data(args.file_path, args.start_date, args.end_date)

def nse_command():
    parser = argparse.ArgumentParser(description="Analyze NSE stocks.")
    parser.add_argument('--num_tickers', type=int, default=None, help="Number of tickers to analyze")
    parser.add_argument('--top_x', type=int, default=None, help="Top X by CAGR")
    parser.add_argument('--start_date', type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end_date', type=str, default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    analyze_nse_stocks(args.num_tickers, args.top_x, args.start_date, args.end_date)

def random_command():
    parser = argparse.ArgumentParser(description="Analyze random stocks or sectors.")
    parser.add_argument('--k', type=int, default=None, help="Number of random selections")
    parser.add_argument('--selection_type', type=str, default="companies", choices=["companies", "sectors"], help="Selection type")
    parser.add_argument('--start_date', type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end_date', type=str, default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    analyze_random_stocks_or_sectors(args.k, args.selection_type, args.start_date, args.end_date)

def indices_command():
    parser = argparse.ArgumentParser(description="Analyze market indices.")
    parser.add_argument('--start_date', type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end_date', type=str, default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    analyze_market_indices(args.start_date, args.end_date)