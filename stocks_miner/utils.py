import os
import pandas as pd
from datetime import datetime, timedelta

def get_default_dates(months_back=6):
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=months_back*30)
    return start_date.isoformat(), end_date.isoformat()

def get_date_input(prompt, default):
    date_str = input(f"{prompt} (default {default}): ").strip()
    if not date_str:
        return default
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        print("Invalid date format! Using default.")
        return default

def get_integer_input(prompt, default=0):
    val = input(f"{prompt} (default {default}): ").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        print("Invalid integer! Using default.")
        return default

def validate_file_path(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist!")
    return file_path

def calculate_cagr(start_price, end_price, days):
    if start_price <= 0 or end_price <= 0 or days <= 0:
        return 0
    return ((end_price / start_price) ** (365 / days) - 1) * 100
