import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from .utils import get_date_input, get_default_dates, get_integer_input

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_market_indices(start_date=None, end_date=None):
    """Analyze SENSEX and NIFTY indices with user-defined highlight periods."""
    # Handle date inputs
    default_start, default_end = get_default_dates()  # Defaults to 6 months prior and today
    start_date = get_date_input("Enter start date", default_start) if start_date is None else pd.to_datetime(start_date)
    end_date = get_date_input("Enter end date", default_end) if end_date is None else pd.to_datetime(end_date)
    
    # Prompt for custom highlight periods
    num_periods = get_integer_input("Enter number of periods to highlight (0 for none)", default=0)
    event_dates = {}
    phases = {}
    if num_periods > 0:
        print("\nEnter details for each highlight period:")
        for i in range(num_periods):
            period_name = input(f"Enter name for period {i+1}: ").strip()
            if not period_name:
                period_name = f"Period {i+1}"
            period_start = get_date_input(f"Enter start date for {period_name}", start_date)
            period_end = get_date_input(f"Enter end date for {period_name}", min(end_date, period_start))
            event_dates[f"{period_name}_start"] = period_start
            event_dates[f"{period_name}_end"] = period_end
            phases[period_name] = (period_start, period_end)
    
    # Download data
    print("Downloading market data...")
    try:
        sensex = yf.download('^BSESN', start=start_date, end=end_date, progress=False)
        nifty = yf.download('^NSEI', start=start_date, end=end_date, progress=False)
        if sensex.empty or nifty.empty:
            raise ValueError("No data downloaded from yfinance.")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        market_data = pd.DataFrame(index=date_range)
        market_data['Sensex'] = sensex['Close'].reindex(date_range, method='ffill')
        market_data['Nifty'] = nifty['Close'].reindex(date_range, method='ffill')
        market_data['Day_Index'] = np.arange(len(date_range))
        print(f"Market data shape: {market_data.shape}")
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise
    
    # Calculate returns and metrics
    market_data['Sensex_Returns'] = market_data['Sensex'].pct_change() * 100
    market_data['Nifty_Returns'] = market_data['Nifty'].pct_change() * 100
    market_data['Sensex_MA20'] = market_data['Sensex'].rolling(window=20).mean()
    market_data['Sensex_MA50'] = market_data['Sensex'].rolling(window=50).mean()
    market_data['Nifty_MA20'] = market_data['Nifty'].rolling(window=20).mean()
    market_data['Nifty_MA50'] = market_data['Nifty'].rolling(window=50).mean()
    market_data['Sensex_Volatility'] = market_data['Sensex_Returns'].rolling(window=20).std()
    market_data['Nifty_Volatility'] = market_data['Nifty_Returns'].rolling(window=20).std()
    
    # Extreme events
    def identify_extreme_events(series, threshold=2.5):
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    market_data['Sensex_Extreme'] = identify_extreme_events(market_data['Sensex_Returns'])
    market_data['Nifty_Extreme'] = identify_extreme_events(market_data['Nifty_Returns'])
    
    # Statistical analysis by phase
    results = {}
    all_returns = {}
    for phase_name, (start, end) in phases.items():
        phase_data = market_data.loc[start:end].copy()
        if len(phase_data) > 1:
            X = phase_data['Day_Index'].values.reshape(-1, 1)
            reg_sensex = LinearRegression().fit(X, phase_data['Sensex'].values)
            reg_nifty = LinearRegression().fit(X, phase_data['Nifty'].values)
            sensex_returns = phase_data['Sensex_Returns'].dropna()
            nifty_returns = phase_data['Nifty_Returns'].dropna()
            all_returns[phase_name] = {'Sensex_Returns': sensex_returns.values, 'Nifty_Returns': nifty_returns.values}
            t_stat_sensex, p_value_sensex = stats.ttest_1samp(sensex_returns, 0)
            t_stat_nifty, p_value_nifty = stats.ttest_1samp(nifty_returns, 0)
            results[phase_name] = {
                'Days': len(phase_data),
                'Sensex_Trend': reg_sensex.coef_[0],
                'Nifty_Trend': reg_nifty.coef_[0],
                'Sensex_Avg_Return': sensex_returns.mean(),
                'Nifty_Avg_Return': nifty_returns.mean(),
                'Sensex_Volatility': sensex_returns.std(),
                'Nifty_Volatility': nifty_returns.std(),
                'Sensex_T_Stat': t_stat_sensex,
                'Sensex_P_Value': p_value_sensex,
                'Nifty_T_Stat': t_stat_nifty,
                'Nifty_P_Value': p_value_nifty,
                'Significant_Sensex': p_value_sensex < 0.05,
                'Significant_Nifty': p_value_nifty < 0.05
            }
    
    # Include overall period if no phases are specified
    if not phases:
        phase_name = "Overall Period"
        phase_data = market_data.copy()
        X = phase_data['Day_Index'].values.reshape(-1, 1)
        reg_sensex = LinearRegression().fit(X, phase_data['Sensex'].values)
        reg_nifty = LinearRegression().fit(X, phase_data['Nifty'].values)
        sensex_returns = phase_data['Sensex_Returns'].dropna()
        nifty_returns = phase_data['Nifty_Returns'].dropna()
        all_returns[phase_name] = {'Sensex_Returns': sensex_returns.values, 'Nifty_Returns': nifty_returns.values}
        t_stat_sensex, p_value_sensex = stats.ttest_1samp(sensex_returns, 0)
        t_stat_nifty, p_value_nifty = stats.ttest_1samp(nifty_returns, 0)
        results[phase_name] = {
            'Days': len(phase_data),
            'Sensex_Trend': reg_sensex.coef_[0],
            'Nifty_Trend': reg_nifty.coef_[0],
            'Sensex_Avg_Return': sensex_returns.mean(),
            'Nifty_Avg_Return': nifty_returns.mean(),
            'Sensex_Volatility': sensex_returns.std(),
            'Nifty_Volatility': nifty_returns.std(),
            'Sensex_T_Stat': t_stat_sensex,
            'Sensex_P_Value': p_value_sensex,
            'Nifty_T_Stat': t_stat_nifty,
            'Nifty_P_Value': p_value_nifty,
            'Significant_Sensex': p_value_sensex < 0.05,
            'Significant_Nifty': p_value_nifty < 0.05
        }
    
    # Create summary DataFrame
    summary_data = []
    for phase, stats_dict in results.items():
        summary_data.append({
            'Period': phase,
            'Days': stats_dict['Days'],
            'Sensex_Trend': stats_dict['Sensex_Trend'],
            'Nifty_Trend': stats_dict['Nifty_Trend'],
            'Sensex_Avg_Return': stats_dict['Sensex_Avg_Return'],
            'Nifty_Avg_Return': stats_dict['Nifty_Avg_Return'],
            'Sensex_Volatility': stats_dict['Sensex_Volatility'],
            'Nifty_Volatility': stats_dict['Nifty_Volatility'],
            'Sensex_Significant': stats_dict['Significant_Sensex'],
            'Nifty_Significant': stats_dict['Significant_Nifty']
        })
    df = pd.DataFrame(summary_data)
    print("\nStatistical Summary:")
    print(df.to_string(index=False))
    
    # Visualizations
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'SENSEX and NIFTY Analysis: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}', fontsize=16, fontweight='bold')
    
    # 1. Price Trends with Events
    ax = axes[0, 0]
    ax.plot(market_data.index, market_data['Sensex'], label='Sensex', color='#FF4B00', linewidth=2)
    ax.plot(market_data.index, market_data['Nifty'], label='Nifty', color='#0055FF', linewidth=2)
    for event, date in event_dates.items():
        ax.axvline(x=date, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.set_ylabel('Index Value')
    ax.set_title('Price Movement with Highlight Markers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Daily Returns
    ax = axes[0, 1]
    ax.plot(market_data.index, market_data['Sensex_Returns'], label='Sensex Returns', color='#FF4B00', alpha=0.7)
    ax.plot(market_data.index, market_data['Nifty_Returns'], label='Nifty Returns', color='#0055FF', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    sensex_extreme_dates = market_data[market_data['Sensex_Extreme']].index
    nifty_extreme_dates = market_data[market_data['Nifty_Extreme']].index
    if len(sensex_extreme_dates) > 0:
        ax.scatter(sensex_extreme_dates, market_data.loc[sensex_extreme_dates, 'Sensex_Returns'],
                   color='red', s=30, label='Sensex Extreme Events', alpha=0.7)
    if len(nifty_extreme_dates) > 0:
        ax.scatter(nifty_extreme_dates, market_data.loc[nifty_extreme_dates, 'Nifty_Returns'],
                   color='blue', s=30, label='Nifty Extreme Events', alpha=0.7)
    ax.set_ylabel('Daily Returns (%)')
    ax.set_title('Daily Returns with Extreme Events')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Trends Comparison
    ax = axes[0, 2]
    periods = df['Period'].values
    x_pos = np.arange(len(periods))
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, df['Sensex_Trend'], width, label='Sensex', color='#FF4B00')
    bars2 = ax.bar(x_pos + width/2, df['Nifty_Trend'], width, label='Nifty', color='#0055FF')
    for i, (sensex_sig, nifty_sig) in enumerate(zip(df['Sensex_Significant'], df['Nifty_Significant'])):
        if sensex_sig:
            ax.text(x_pos[i] - width/2, df['Sensex_Trend'].iloc[i] + (10 if df['Sensex_Trend'].iloc[i] > 0 else -20),
                   '*', ha='center', va='center', fontweight='bold', fontsize=12)
        if nifty_sig:
            ax.text(x_pos[i] + width/2, df['Nifty_Trend'].iloc[i] + (3 if df['Nifty_Trend'].iloc[i] > 0 else -6),
                   '*', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.set_xlabel('Periods')
    ax.set_ylabel('Trend (points/day)')
    ax.set_title('Daily Trends with Significance (*)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Average Returns Comparison
    ax = axes[1, 0]
    bars1 = ax.bar(x_pos - width/2, df['Sensex_Avg_Return'], width, label='Sensex', color='#FF4B00')
    bars2 = ax.bar(x_pos + width/2, df['Nifty_Avg_Return'], width, label='Nifty', color='#0055FF')
    for i, (sensex_sig, nifty_sig) in enumerate(zip(df['Sensex_Significant'], df['Nifty_Significant'])):
        if sensex_sig:
            ax.text(x_pos[i] - width/2, df['Sensex_Avg_Return'].iloc[i] + (0.01 if df['Sensex_Avg_Return'].iloc[i] > 0 else -0.03),
                   '*', ha='center', va='center', fontweight='bold', fontsize=12)
        if nifty_sig:
            ax.text(x_pos[i] + width/2, df['Nifty_Avg_Return'].iloc[i] + (0.01 if df['Nifty_Avg_Return'].iloc[i] > 0 else -0.03),
                   '*', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.set_xlabel('Periods')
    ax.set_ylabel('Average Daily Return (%)')
    ax.set_title('Average Daily Returns with Significance')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Volatility Comparison
    ax = axes[1, 1]
    bars1 = ax.bar(x_pos - width/2, df['Sensex_Volatility'], width, label='Sensex', color='#FF4B00')
    bars2 = ax.bar(x_pos + width/2, df['Nifty_Volatility'], width, label='Nifty', color='#0055FF')
    ax.set_xlabel('Periods')
    ax.set_ylabel('Volatility (%)')
    ax.set_title('Volatility Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Risk-Return Scatter
    ax = axes[1, 2]
    scatter = ax.scatter(df['Sensex_Avg_Return'], df['Sensex_Volatility'], s=100,
                        c=range(len(df)), cmap='viridis', alpha=0.7)
    for i, period in enumerate(periods):
        ax.annotate(period, (df['Sensex_Avg_Return'].iloc[i], df['Sensex_Volatility'].iloc[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Average Return (%)')
    ax.set_ylabel('Volatility (%)')
    ax.set_title('Risk-Return Profile (Sensex)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=df['Sensex_Volatility'].mean(), color='red', linestyle='--', alpha=0.5, label='Avg Volatility')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Correlation Heatmap
    ax = axes[2, 0]
    correlation_matrix = market_data[['Sensex_Returns', 'Nifty_Returns']].corr()
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Sensex Returns', 'Nifty Returns'])
    ax.set_yticklabels(['Sensex Returns', 'Nifty Returns'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}',
                   ha='center', va='center', fontweight='bold', color='white')
    ax.set_title('Returns Correlation Heatmap')
    plt.colorbar(im, ax=ax)
    
    # 8. Boxplot of Returns
    ax = axes[2, 1]
    returns_data = [all_returns[phase]['Sensex_Returns'] for phase in all_returns.keys()]
    labels = list(all_returns.keys())
    ax.boxplot(returns_data, labels=labels)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Daily Returns (%)')
    ax.set_title('Distribution of Sensex Returns by Period')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 9. Cumulative Returns
    ax = axes[2, 2]
    market_data['Sensex_Cumulative'] = (1 + market_data['Sensex_Returns']/100).cumprod()
    market_data['Nifty_Cumulative'] = (1 + market_data['Nifty_Returns']/100).cumprod()
    ax.plot(market_data.index, market_data['Sensex_Cumulative'], label='Sensex', color='#FF4B00')
    ax.plot(market_data.index, market_data['Nifty_Cumulative'], label='Nifty', color='#0055FF')
    ax.set_ylabel('Cumulative Return (Index = 1)')
    ax.set_title('Cumulative Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    # Detailed Statistical Tests
    print("\n" + "=" * 60)
    print("DETAILED STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)
    for phase in all_returns.keys():
        stats_data = results[phase]
        print(f"\n{phase}:")
        print(f" Sensex - T-stat: {stats_data['Sensex_T_Stat']:.3f}, P-value: {stats_data['Sensex_P_Value']:.4f}")
        print(f" Nifty - T-stat: {stats_data['Nifty_T_Stat']:.3f}, P-value: {stats_data['Nifty_P_Value']:.4f}")
        if stats_data['Significant_Sensex']:
            direction = "positive" if stats_data['Sensex_Avg_Return'] > 0 else "negative"
            print(f" ✓ Sensex shows statistically significant {direction} trend")
        if stats_data['Significant_Nifty']:
            direction = "positive" if stats_data['Nifty_Avg_Return'] > 0 else "negative"
            print(f" ✓ Nifty shows statistically significant {direction} trend")
    
    # ANOVA test
    sensex_returns_groups = [all_returns[phase]['Sensex_Returns'] for phase in all_returns.keys()]
    if len(sensex_returns_groups) > 1:
        f_stat, p_value = stats.f_oneway(*sensex_returns_groups)
        print(f"\nANOVA Test for Sensex Returns across periods:")
        print(f" F-statistic: {f_stat:.3f}, P-value: {p_value:.4f}")
        if p_value < 0.05:
            print(" ✓ Statistically significant difference in returns across periods")
    
    # Correlation analysis
    correlation = stats.pearsonr(market_data['Sensex_Returns'].dropna(), market_data['Nifty_Returns'].dropna())
    print(f"\nCorrelation between Sensex and Nifty returns:")
    print(f" Correlation coefficient: {correlation[0]:.4f}")
    print(f" P-value: {correlation[1]:.4f}")
    
    return df, results, all_returns