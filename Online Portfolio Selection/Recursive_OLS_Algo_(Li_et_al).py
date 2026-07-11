import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
import itertools
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# 0. Data fetching
def get_price(ret_df, decision_date, window=30):
    if not isinstance(decision_date, pd.Timestamp):
        decision_date = pd.Timestamp(decision_date)
    try:
        hist_returns = ret_df.loc[:decision_date]
        data_x = hist_returns.iloc[-window:]
        data_y = hist_returns.iloc[-window-1:-1]
        x = hist_returns.iloc[-1]
        y = hist_returns.iloc[-2]
        return data_x.to_numpy().T, data_y.to_numpy().T, x.to_numpy().T, y.to_numpy().T
    except Exception:
        return None


# 1. Initialization
def get_parameter_A_P_gma(data_x, data_y):
    m = data_x.shape[0]
    Q = data_x @ data_y.T
    P = np.linalg.pinv(data_x @ data_x.T + np.eye(m) * 1e-3)
    A = Q @ P
    return A, P


# 2. Portfolio Update
def get_weight(current_weight, A, x, eta=0.05):
    Ax = A @ x
    denom = current_weight @ Ax + 1e-12
    utility = Ax / denom
    exponent = eta * utility
    exponent -= np.max(exponent)
    num = current_weight * np.exp(exponent)
    target_weight = num / (np.sum(num) + 1e-12)
    return target_weight


# 3. Transaction Costs
def find_transaction_loss(current_weight, target_weight, fee_proportion=0.001):
    l1 = np.abs(current_weight - target_weight).sum()
    w = 1.0 / (1.0 + fee_proportion * l1)
    return max(w, 0.0)


# 4. Wealth Update
def cumulative_wealth(position, transaction_loss, target_weight, realized_returns):
    return position * transaction_loss * (target_weight @ realized_returns)


# 5. Weight Drift Update
def update_weight(old_target_weight, realized_returns):
    denom = old_target_weight @ realized_returns + 1e-12
    return (old_target_weight * realized_returns) / denom


# 6. Recursive DMD Update
def update_parameter_A_P_gma(x, y, A, P, sig, t_step):
    denom = (1 + y.T @ P @ y)
    gma = 1 / (denom if denom != 0 else 1e-12)
    x_hat_diff = x - A @ y
    A = A + gma * np.outer(x_hat_diff, y @ P)
    log_factor = np.emath.logn(t_step + 1, t_step + 2)
    weight = (log_factor if log_factor > 0 else 1.0) ** sig
    P = weight * (P - gma * np.outer(P @ y, y @ P))
    return A, P


# Main Simulation
def run_li_sim(price_df, decision_date, total_days=200, window=30, eta=0.05, sig=1.0, fee_proportion=0.002):
    np.seterr(all='ignore')
    position = 1.0
    n_assets = price_df.shape[1]
    current_weight = np.full(n_assets, 1.0 / n_assets)
    wealth_history = [1.0]
    decision_date = pd.Timestamp(decision_date)
    ret_df = 1.0 + price_df.pct_change().dropna(how='any')
    trading_days = ret_df.loc[decision_date:].index[:total_days + 1]
    for i in range(len(trading_days) - 1):
        current_date = trading_days[i]
        next_date = trading_days[i+1]
        price_data = get_price(ret_df, current_date, window)
        if price_data is None:
            wealth_history.append(position)
            continue
        data_x, data_y, x, y = price_data
        if i == 0:
            A, P = get_parameter_A_P_gma(data_x, data_y)
        else:
            t_step = window + i
            A, P = update_parameter_A_P_gma(x, y, A, P, sig, t_step)
        target_weight = get_weight(current_weight, A, x, eta)
        w = find_transaction_loss(current_weight, target_weight, fee_proportion)
        realized_r = ret_df.loc[next_date].values
        position = cumulative_wealth(position, w, target_weight, realized_r)
        if np.isnan(position) or position <= 0:
            return np.nan
        current_weight = update_weight(target_weight, realized_r)
        wealth_history.append(position)
    return wealth_history[-1]


if __name__ == "__main__":
    tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'PG', 'KO', 'PEP', 'XOM', 'CVX',
        'JPM', 'BAC', 'WFC', 'GS', 'HD', 'MCD', 'NKE', 'DIS', 'VZ', 'T',
        'PFE', 'MRK', 'ABBV', 'LLY', 'UNH', 'CVS', 'WMT', 'COST', 'PM', 'MO',
        'IBM', 'ORCL', 'CSCO', 'INTC', 'QCOM', 'TXN', 'AVGO', 'CAT', 'DE',
        'BA', 'HON', 'GE', 'MMM', 'DOW', 'EMN', 'APD', 'NEE', 'DUK', 'SO'
    ]

    price_df = yf.download(tickers=tickers, start='2015-07-01', end='2026-07-01', progress=False, auto_adjust=True)['Close']
    price_df.index = price_df.index.tz_localize(None)
    price_df = price_df.dropna(how='any')
    eta_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    sigma_values = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    window_values = [10, 20, 30, 60, 90, 120]
    fee = 0.002
    param_grid = list(itertools.product(eta_values, sigma_values, window_values))
    experts_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                run_li_sim,
                price_df,
                '2022-08-21',
                500,
                wnd,
                et,
                sg,
                fee
            ): (et, sg, wnd)
            for et, sg, wnd in param_grid
        }
        for future in concurrent.futures.as_completed(futures):
            et, sg, wnd = futures[future]
            try:
                final_wealth = future.result()
                if pd.notna(final_wealth):
                    experts_list.append({
                        'eta': round(et, 5),
                        'sigma': round(sg, 3),
                        'window': wnd,
                        'final_wealth': final_wealth
                    })
            except Exception:
                pass
    expert_df = pd.DataFrame(experts_list)
    if not expert_df.empty:
        print(expert_df.sort_values('final_wealth', ascending=False).head(30).to_string(index=False))
    else:
        print("All combinations resulted in NaN/Overflow.")
