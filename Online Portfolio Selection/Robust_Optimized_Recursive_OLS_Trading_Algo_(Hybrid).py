import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import concurrent.futures
import itertools
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_parameter_A_P(data_x, data_y):
    m = data_x.shape[0]
    Q = data_x @ data_y.T
    P = np.linalg.pinv(data_x @ data_x.T + np.eye(m) * 1e-3)
    A = Q @ P
    return A, P


def update_parameter_A_P(x_next, x_t, A, P, sig, t_step):
    denom = (1 + x_t.T @ P @ x_t)
    gma = 1 / (denom if denom != 0 else 1e-12)
    x_hat_diff = x_next - A @ x_t
    A = A + gma * np.outer(x_hat_diff, x_t @ P)
    log_factor = np.emath.logn(t_step + 1, t_step + 2)
    weight = (log_factor if log_factor > 0 else 1.0) ** sig
    P = weight * (P - gma * np.outer(P @ x_t, P @ x_t))
    return A, P


def get_cov(data):
    if data is None or len(data) < 2:
        n = data.shape[1] if data is not None else 1
        return np.eye(n) * 0.0001
    return data.pct_change().dropna().cov().values


def optimize_weight(cov_matrix, current_weight, predicted_returns, lmda, kappa, fee_proportion=0.002):
    n = len(predicted_returns)
    try:
        L = np.linalg.cholesky(cov_matrix)
        U = L.T
    except np.linalg.LinAlgError:
        U = np.eye(n) * 0.01
    def objective(b):
        port_return = b @ predicted_returns
        l1_penalty = lmda * np.linalg.norm(current_weight - b, ord=1)
        variance_reward = kappa * np.linalg.norm(U @ b, ord=2)
        return -(port_return - l1_penalty + variance_reward)
    def constraint(b):
        return 1.0 - (np.sum(b) + fee_proportion * np.linalg.norm(current_weight - b, ord=1))
    constraints = [{'type': 'ineq', 'fun': constraint}]
    bounds = [(0.0, 1.0)] * n
    initial_guess = current_weight * 0.99
    result = minimize(
        fun=objective,
        x0=initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-4, 'maxiter': 50}
    )
    if result.success:
        raw_b = np.maximum(result.x, 0.0)
        w = np.sum(raw_b)
        target_weight = raw_b / (w + 1e-12)
        return target_weight, w
    else:
        return current_weight.copy(), 1.0


def run_hybrid_sim(price_df, decision_date, total_days=200,
                   dmd_window=60, cov_window=51,
                   lmda=0.02, kappa=1.0, sig=0.1, fee_proportion=0.002):
    np.seterr(all='ignore')
    position = 1.0
    n_assets = price_df.shape[1]
    current_weight = np.full(n_assets, 1.0 / n_assets)
    wealth_history = [1.0]
    ret_df = 1.0 + price_df.pct_change().dropna(how='any')
    decision_date = pd.Timestamp(decision_date)
    trading_days = ret_df.loc[decision_date:].index[:total_days + 1]
    start_idx = ret_df.index.get_loc(trading_days[0])
    hist = ret_df.iloc[start_idx - dmd_window : start_idx].values
    X_dmd = hist[:-1].T
    Y_dmd = hist[1:].T
    A, P = get_parameter_A_P(X_dmd, Y_dmd)
    x_t = hist[-1]
    for i in range(len(trading_days) - 1):
        current_date = trading_days[i]
        next_date = trading_days[i+1]
        predicted_returns = A @ x_t
        cov_data = price_df.loc[:current_date].iloc[-cov_window:]
        cov_matrix = get_cov(cov_data)
        target_weight, w = optimize_weight(
            cov_matrix=cov_matrix,
            current_weight=current_weight,
            predicted_returns=predicted_returns,
            lmda=lmda,
            kappa=kappa,
            fee_proportion=fee_proportion
        )
        x_next = ret_df.loc[next_date].values
        position = position * w * (target_weight @ x_next)
        if np.isnan(position) or position <= 0:
            return np.nan
        current_weight = (target_weight * x_next) / (np.dot(target_weight, x_next) + 1e-12)
        wealth_history.append(position)
        t_step = dmd_window + i
        A, P = update_parameter_A_P(x_next, x_t, A, P, sig, t_step)
        x_t = x_next
    return wealth_history[-1]


if __name__ == "__main__":
    tickers = [
        'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AMZN', 'AVGO', 'BAC', 'BRK-B', 'CMCSA',
        'COST', 'CRM', 'CSCO', 'CVX', 'DHR', 'DIS', 'META', 'GOOG', 'GOOGL', 'HD',
        'INTC', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'MA', 'MCD', 'MDT', 'MRK',
        'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PYPL',
        'T', 'TMO', 'TSLA', 'TXN', 'UNH', 'V', 'VZ', 'WFC', 'WMT', 'XOM'
    ]
    price_df = yf.download(tickers=tickers, start='2015-01-01', end='2026-07-01', progress=False, auto_adjust=True)['Close']
    price_df.index = price_df.index.tz_localize(None)
    price_df = price_df.dropna(how='any')
    fee = 0.002
    cov_wnd = len(tickers) + 1
    lmda_values = [1 * fee, 10 * fee]
    kappa_values = [0.1, 0.5, 1.0, 2.0]
    sigma_values = [0.0, 0.1, 0.5]
    dmd_values = [60]
    param_grid = list(itertools.product(lmda_values, kappa_values, sigma_values, dmd_values))
    experts_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                run_hybrid_sim,
                price_df,
                '2024-05-01',
                250,
                d_wnd,
                cov_wnd,
                lm,
                kp,
                sg,
                fee
            ): (lm, kp, sg, d_wnd)
            for lm, kp, sg, d_wnd in param_grid
        }
        for future in concurrent.futures.as_completed(futures):
            lm, kp, sg, d_wnd = futures[future]
            try:
                final_wealth = future.result()
                if pd.notna(final_wealth):
                    experts_list.append({
                        'lambda': lm,
                        'kappa': kp,
                        'sigma': sg,
                        'final_wealth': final_wealth
                    })
            except Exception:
                pass
    expert_df = pd.DataFrame(experts_list)
    if not expert_df.empty:
        print(expert_df.sort_values('final_wealth', ascending=False).head(25).to_string(index=False))
    else:
        print("All combinations failed.")
