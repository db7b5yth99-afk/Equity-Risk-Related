import os
import pandas as pd
import yfinance as yf


def get_market_data(tickers, start_date, end_date):
    """
    Downloads and caches OHLCV data for the designated universe.
    Returns Open, High, Low, Close, and Volume DataFrames.
    """
    cache_file = f"{start_date}_to_{end_date}_stock_data_ohlcv.csv"

    if os.path.exists(cache_file):
        print(f"Loading cached OHLCV data from {cache_file}...")
        df = pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)

        open_df = df['Open'].ffill() if 'Open' in df.columns.levels[0] else df.ffill()
        high_df = df['High'].ffill() if 'High' in df.columns.levels[0] else df.ffill()
        low_df = df['Low'].ffill() if 'Low' in df.columns.levels[0] else df.ffill()
        close_df = df['Close'].ffill() if 'Close' in df.columns.levels[0] else df.ffill()
        vol_df = df['Volume'].ffill() if 'Volume' in df.columns.levels[0] else df.ffill()
    else:
        print("Downloading full universe OHLCV data from Yahoo Finance...")
        df = yf.download(tickers, start=start_date, end=end_date, progress=False)
        df.ffill(inplace=True)
        df.to_csv(cache_file)

        open_df = df['Open'].ffill()
        high_df = df['High'].ffill()
        low_df = df['Low'].ffill()
        close_df = df['Close'].ffill()
        vol_df = df['Volume'].ffill()

    # Timezone standardization
    for d in [open_df, high_df, low_df, close_df, vol_df]:
        d.index = d.index.tz_localize(None)

    return open_df[tickers], high_df[tickers], low_df[tickers], close_df[tickers], vol_df[tickers]


def get_data_vector(df, end_date, window, include_current=False):
    """
    Retrieves a window of data up to a specified end_date.
    Setting include_current=False ensures strictly historical data is fetched,
    preventing look-ahead bias (data leaks).
    """
    end_date = pd.to_datetime(end_date)

    if include_current:
        df_mod = df.loc[df.index <= end_date]
    else:
        df_mod = df.loc[df.index < end_date]

    return df_mod.iloc[-window:]


if __name__ == "__main__":
    print("Data Ingestion Module loaded successfully.")
