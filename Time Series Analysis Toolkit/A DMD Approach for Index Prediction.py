import yfinance as yf
import numpy as np
import pandas as pd

tickers = [
    "^GSPC",                                     # Target
    "XLK", "XLF", "XLV", "XLY", "XLI", "XLE", "XLB", "XLU", # Sectors
    "MTUM", "VLUE", "QUAL", "SPLV", "IWM",     # Smart Beta Factors
    "TLT", "IEF", "GLD", "UUP", "USO"          # Cross-Asset Macro
]
price = yf.download(tickers=tickers,start='2020-01-01')['Close']
pct = 1 + price.pct_change().dropna(how='any')

def dmd_pred(df, t=None):
    if t is None:
        t=252*5
        
    window = df.iloc[-t-1:]
    mean = window.mean()
    std = window.std()
    df5y = (df.iloc[-t:] - mean) / std
    df5y_utility = (df.iloc[-t-1:-1] - mean) / std
    X = df5y_utility.to_numpy().T
    Y = df5y.to_numpy().T
    x_latest = Y[:, 0]
    X_inverse = np.linalg.pinv(X)
    
    A = Y @ X_inverse
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    A_tilde = U.T @ A @ U
    lmd, W = np.linalg.eig(A_tilde)
    phi = Y @ Vt.T @ np.diag(1.0 / Sigma) @ W
    b = np.linalg.pinv(phi) @ x_latest
    X_pred = phi @ np.diag(lmd) @ b    
    X_pred = (X_pred * std.to_numpy()) + mean.to_numpy()
    spy_idx = list(df.columns).index("^GSPC")
    
    return X_pred[spy_idx].real



print(f"S&P 500 Last Close Price: {price['^GSPC'].iloc[-1]:.2f}")
print('DMD Prediction of S&P 500 in 1-d')
ret1 = price['^GSPC'].iloc[-1] * dmd_pred(df=pct, t=5)
ret2 = price['^GSPC'].iloc[-1] * dmd_pred(df=pct, t=21)
ret3 = price['^GSPC'].iloc[-1] * dmd_pred(df=pct, t=63)
ret4 = price['^GSPC'].iloc[-1] * dmd_pred(df=pct, t=126)
ret5 = price['^GSPC'].iloc[-1] * dmd_pred(df=pct, t=252)
ret6 = price['^GSPC'].iloc[-1] * dmd_pred(df=pct, t=252*5)
ret_avg = np.mean([ret1,ret2,ret3,ret4,ret5,ret6])
print(f'5-d Data Predicted Price: {ret1:.2f}')
print(f'1-m Data Predicted Price: {ret2:.2f}')
print(f'3-m Data Predicted Price: {ret3:.2f}')
print(f'6-m Data Predicted Price: {ret4:.2f}')
print(f'1-y Data Predicted Price: {ret5:.2f}')
print(f'5-y Data Predicted Price: {ret6:.2f}')
print(f'\nAverage Predicted Price: {ret_avg:.2f}')
