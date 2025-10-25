import datetime
import os
import pandas as pd

def get_default_dates(months_back=6):
    """Get default start (6 months ago) and end (today) dates."""
    today = datetime.date.today()
    start = today - datetime.timedelta(days=30 * months_back)
    return start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')

def get_date_input(prompt, default):
    """Prompt for date input with default."""
    user_input = input(f"{prompt} (YYYY-MM-DD, default: {default}): ").strip()
    if not user_input:
        return pd.to_datetime(default)
    try:
        return pd.to_datetime(user_input)
    except ValueError:
        print("Invalid date format. Using default.")
        return pd.to_datetime(default)

def get_integer_input(prompt, default=0):
    """Prompt for integer input with default."""
    user_input = input(f"{prompt} (default: {default}): ").strip()
    if not user_input:
        return default
    try:
        return int(user_input)
    except ValueError:
        print("Invalid integer. Using default.")
        return default

def validate_file_path(file_path):
    """Validate file path exists; reprompt if not."""
    while not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        file_path = input("Enter valid path to your .xlsx or .csv file: ").strip()
    return file_path

def calculate_cagr(start_price, end_price, days):
    """Calculate Compound Annual Growth Rate (CAGR)."""
    if start_price == 0 or days <= 0:
        return 0.0
    years = days / 365.25
    return ((end_price / start_price) ** (1 / years)) - 1