import requests
import pandas as pd
import io
import yfinance as yf
from yahooquery import Ticker
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from .utils import get_date_input, get_default_dates, get_integer_input, calculate_cagr

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_random_stocks_or_sectors(k=None, selection_type="companies", start_date=None, end_date=None):
    """Analyze k randomly selected companies or sectors with statistical metrics."""
    start_time = time.time()
    
    # Handle inputs
    k = get_integer_input("Enter number of random selections (k)", default=5) if k is None else k
    if selection_type not in ["companies", "sectors"]:
        selection_type = input("Enter selection type ('companies' or 'sectors', press Enter for 'companies'): ").strip() or "companies"
    default_start, default_end = get_default_dates()
    start_date = get_date_input("Enter start date", default_start) if start_date is None else pd.to_datetime(start_date)
    end_date = get_date_input("Enter end date", default_end) if end_date is None else pd.to_datetime(end_date)
    
    # Download NSE stock list
    print("📡 Downloading NSE stock list...")
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    df_nse = pd.read_csv(io.BytesIO(response.content))
    df_nse["Yahoo_Ticker"] = df_nse["SYMBOL"].astype(str) + ".NS"
    df_nse = df_nse.reset_index(drop=True)
    print(f"✅ Total NSE stocks loaded: {len(df_nse)}")
    
    # Fetch asset profile info for all stocks
    print("🔍 Fetching asset profile information...")
    tickers = df_nse["Yahoo_Ticker"].tolist()
    t = Ticker(tickers)
    infos = t.asset_profile
    
    # Map asset profile data
    asset_profile_fields = set()
    for symbol, data in infos.items():
        if isinstance(data, dict):
            asset_profile_fields.update(data.keys())
    for field in asset_profile_fields:
        df_nse[field] = "Unknown"
    for symbol, data in infos.items():
        if isinstance(data, dict):
            for field in asset_profile_fields:
                df_nse.loc[df_nse["Yahoo_Ticker"] == symbol, field] = str(data.get(field, "Unknown"))
    
    # Random selection
    if selection_type == "sectors":
        unique_sectors = df_nse['sector'].unique()
        unique_sectors = [s for s in unique_sectors if s != "Unknown"]
        if len(unique_sectors) < k:
            print(f"Warning: Only {len(unique_sectors)} sectors available. Selecting all.")
            selected = unique_sectors
        else:
            selected = np.random.choice(unique_sectors, k, replace=False)
        print(f"Randomly selected sectors: {', '.join(selected)}")
        selected_tickers = df_nse[df_nse['sector'].isin(selected)]['Yahoo_Ticker'].tolist()
    else:
        selected_tickers = np.random.choice(tickers, min(k, len(tickers)), replace=False)
        print(f"Randomly selected companies: {', '.join(selected_tickers)}")
    
    # Data analysis
    print("\n📊 Starting data analysis...")
    summary_list = []
    for ticker in tqdm(selected_tickers, desc="Processing tickers"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            time.sleep(0.05)
            if data.empty:
                print(f" ❌ No data for {ticker}")
                continue
            data = data.reset_index()
            data['Returns'] = data['Close'].pct_change() * 100
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            
            # Linear regression
            if len(data) > 1:
                X = np.arange(len(data)).reshape(-1, 1)
                y = data['Close'].values.reshape(-1, 1)
                reg = LinearRegression().fit(X, y)
                trend_slope = reg.coef_[0][0]
                r_squared = reg.score(X, y)
            else:
                trend_slope = r_squared = np.nan
            
            # T-test
            returns = data['Returns'].dropna()
            t_stat, p_value = stats.ttest_1samp(returns, 0) if len(returns) > 1 else (np.nan, np.nan)
            
            # Statistics
            avg_return = returns.mean() if len(returns) > 0 else np.nan
            volatility = returns.std() if len(returns) > 1 else np.nan
            sharpe_ratio = np.sqrt(252) * (avg_return / 100 / volatility) if volatility > 0 else np.nan
            
            # CAGR
            days = (data['Date'].iloc[-1] - data['Date'].iloc[0]).days
            cagr = calculate_cagr(data['Close'].iloc[0], data['Close'].iloc[-1], days) * 100
            
            # Company info
            company_row = df_nse[df_nse["Yahoo_Ticker"] == ticker]
            company_name = company_row["NAME OF COMPANY"].iloc[0] if not company_row.empty else "Unknown"
            sector = company_row.get('sector', pd.Series(["Unknown"])).iloc[0]
            
            summary_list.append({
                "Ticker": ticker,
                "Company_Name": company_name,
                "Sector": sector,
                "Avg_Return_%": round(avg_return, 4),
                "Volatility_%": round(volatility, 4),
                "Trend_Slope": round(trend_slope, 2),
                "R_Squared": round(r_squared, 4),
                "T_Stat": round(t_stat, 4),
                "P_Value": round(p_value, 4),
                "Sharpe_Ratio": round(sharpe_ratio, 4),
                "CAGR_%": round(cagr, 4) if not np.isnan(cagr) else np.nan,
                "Significant_Trend": p_value < 0.05 if not np.isnan(p_value) else False
            })
            print(f" ✅ {ticker}: Avg Return = {avg_return:.2f}%, Volatility = {volatility:.2f}%, Trend Slope = {trend_slope:.2f}, CAGR = {cagr:.2f}%")
        except Exception as e:
            print(f" ❌ Error processing {ticker}: {str(e)}")
    
    print(f"\n🎯 Analysis Complete! Time: {time.time() - start_time:.2f} seconds")
    
    if summary_list:
        final_summary = pd.DataFrame(summary_list)
        if final_summary['Ticker'].duplicated().any():
            final_summary = final_summary.drop_duplicates(subset=['Ticker']).reset_index(drop=True)
        
        # Company-wise analysis
        print("\n📋 SELECTED STATISTICS:")
        print(final_summary[['Ticker', 'Company_Name', 'Sector', 'Avg_Return_%', 'Volatility_%', 'Trend_Slope', 'CAGR_%', 'Sharpe_Ratio']].to_string(index=False))
        
        # Top k performers by CAGR
        top_cagr = final_summary.nlargest(k, 'CAGR_%')[['Ticker', 'Company_Name', 'Sector', 'Avg_Return_%', 'Volatility_%', 'CAGR_%', 'Sharpe_Ratio']]
        print(f"\n🏆 TOP {k} PERFORMERS (by CAGR):")
        print(top_cagr.to_string(index=False))
        
        # Overall statistics
        overall_stats = final_summary[['Avg_Return_%', 'Volatility_%', 'Trend_Slope', 'Sharpe_Ratio', 'CAGR_%', 'Significant_Trend']].describe().round(4)
        print("\n📊 OVERALL STATISTICS:")
        print(overall_stats)
        
        # Correlation
        corr_matrix = final_summary[['Avg_Return_%', 'Volatility_%', 'Trend_Slope', 'Sharpe_Ratio', 'CAGR_%']].corr()
        print("\n🔗 CORRELATION MATRIX:")
        print(corr_matrix)
        
        # Visualizations
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=final_summary, x='Volatility_%', y='Avg_Return_%', hue='Sector', size='CAGR_%', sizes=(50, 200))
        plt.title('Average Returns vs Volatility (Size by CAGR)')
        plt.xlabel('Volatility (%)')
        plt.ylabel('Average Return (%)')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=final_summary['Volatility_%'].mean(), color='red', linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=final_summary['Ticker'], y=final_summary['Sharpe_Ratio'], hue=final_summary['Sector'])
        plt.title('Sharpe Ratio by Selection')
        plt.xlabel('Ticker')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Trend-return correlation
        trend_return_corr = stats.pearsonr(final_summary['Trend_Slope'].dropna(), final_summary['Avg_Return_%'].dropna())
        print(f"\n🔗 Correlation between Trend Slope and Returns:")
        print(f"Correlation coefficient: {trend_return_corr[0]:.4f}, P-value: {trend_return_corr[1]:.4f}")
        
        return final_summary, overall_stats, corr_matrix, top_cagr
    return None, None, None, None