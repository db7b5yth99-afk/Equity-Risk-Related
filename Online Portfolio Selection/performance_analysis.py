import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def calculate_metrics(wealth_series, risk_free_rate=0.02):
    """
    Calculates common institutional performance metrics for a given wealth series.
    Assumes 252 trading days per year.
    """
    if not isinstance(wealth_series.index, pd.DatetimeIndex):
        wealth_series.index = pd.to_datetime(wealth_series.index)

    wealth_series = wealth_series.dropna()
    if len(wealth_series) < 2:
        return {}

    daily_returns = wealth_series.pct_change().dropna()

    trading_days = len(wealth_series)
    years = trading_days / 252.0

    total_return = wealth_series.iloc[-1] / wealth_series.iloc[0]
    cagr = (total_return ** (1 / years)) - 1.0

    volatility = daily_returns.std() * np.sqrt(252)

    rolling_max = wealth_series.cummax()
    drawdown = (wealth_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    if volatility > 0:
        sharpe_ratio = (cagr - risk_free_rate) / volatility
    else:
        sharpe_ratio = 0.0

    downside_returns = daily_returns[daily_returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    if downside_volatility > 0:
        sortino_ratio = (cagr - risk_free_rate) / downside_volatility
    else:
        sortino_ratio = 0.0

    if abs(max_drawdown) > 0:
        calmar_ratio = cagr / abs(max_drawdown)
    else:
        calmar_ratio = np.nan

    return {
        'Total Return (%)': (total_return - 1.0) * 100,
        'YoY Return (CAGR) (%)': cagr * 100,
        'Annualized Volatility (%)': volatility * 100,
        'Max Drawdown (%)': max_drawdown * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio
    }


def plot_performance(wealth_data_dict, index_tickers=['^HSI'], start_date=None, end_date=None,
                     save_path="performance_plot.png"):
    """
    Plots individual or combined wealth history series against specified index/indices.
    Displays directly in PyCharm using plt.show().
    """
    plt.figure(figsize=(16, 8))

    first_series = list(wealth_data_dict.values())[0]
    if not isinstance(first_series.index, pd.DatetimeIndex):
        first_series.index = pd.to_datetime(first_series.index)

    if not start_date:
        start_date = first_series.index.min().strftime('%Y-%m-%d')
    if not end_date:
        end_date = first_series.index.max().strftime('%Y-%m-%d')

    baseline_dates = first_series.loc[start_date:end_date].index
    baseline_initial_wealth = first_series.loc[start_date:end_date].iloc[0]

    for ticker in index_tickers:
        cache_file = f"index_cache_{ticker.replace('^', '')}_{start_date}_to_{end_date}.csv"

        if os.path.exists(cache_file):
            index_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            print(f"Downloading {ticker} data from Yahoo Finance...")
            index_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            index_df.to_csv(cache_file)

        if 'Close' in index_df.columns:
            if isinstance(index_df.columns, pd.MultiIndex):
                close_prices = index_df['Close'][ticker].ffill()
            else:
                close_prices = index_df['Close'].ffill()

            close_prices = close_prices.reindex(baseline_dates).ffill().bfill()
            normalized_index = (close_prices / close_prices.iloc[0]) * baseline_initial_wealth

            plt.plot(normalized_index.index, normalized_index, label=f'Benchmark: {ticker}',
                     linestyle='--', linewidth=2.5, color='black', zorder=1)

    for name, series in wealth_data_dict.items():
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)

        aligned_series = series.reindex(baseline_dates).ffill()
        normalized_series = (aligned_series / aligned_series.iloc[0]) * baseline_initial_wealth

        plt.plot(normalized_series.index, normalized_series, label=f'Strategy: {name}', linewidth=1.8, zorder=2)

    plt.title('Strategy Performance vs Benchmarks', fontsize=16)
    plt.ylabel('Total Wealth (Normalized to Initial Capital)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Performance plot saved to {save_path}")

    # Render interactive plot directly in PyCharm
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("Performance Analysis Module loaded successfully.")