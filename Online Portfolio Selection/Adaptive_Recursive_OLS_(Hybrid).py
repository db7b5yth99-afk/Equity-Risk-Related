import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import itertools
import concurrent.futures
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1. DMD Momentum Prediction Functions (Stateful)
# ============================================================

def get_parameter_A_P_gma(data_x, data_y):
    m = data_x.shape[0]
    Q = data_x @ data_y.T
    P = np.linalg.pinv(data_x @ data_x.T + np.eye(m) * 1e-3)
    A = Q @ P
    return A, P


def update_parameter_A_P_gma(x, y, A, P, sig, t_step):
    denom = (1 + y.T @ P @ y)
    gma = 1 / (denom if denom != 0 else 1e-12)
    x_hat_diff = x - A @ y
    A = A + gma * np.outer(x_hat_diff, y @ P)
    log_factor = np.emath.logn(t_step + 1, t_step + 2)
    weight = (log_factor if log_factor > 0 else 1.0) ** sig
    P = weight * (P - gma * np.outer(P @ y, y @ P))
    return A, P


# ============================================================
# 2. Robust Optimization Functions (PATCHED)
# ============================================================

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
        risk_penalty = kappa * np.linalg.norm(U @ b, ord=2)
        return -(port_return - l1_penalty - risk_penalty)

    def constraint(b):
        return 1.0 - (np.sum(b) + fee_proportion * np.linalg.norm(current_weight - b, ord=1))

    constraints = [{'type': 'eq', 'fun': constraint}]
    bounds = [(0.0, 1.0)] * n

    initial_guess = np.full(n, (1.0 - fee_proportion) / n)

    result = minimize(fun=objective, x0=initial_guess, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-4, 'maxiter': 50})

    if result.success:
        raw_b = np.maximum(result.x, 0.0)
        w = np.sum(raw_b)
        if w < 1e-6:
            return current_weight.copy(), 1.0
        target_weight = raw_b / w
        return target_weight, w
    else:
        return current_weight.copy(), 1.0


def update_portfolio(cov_matrix, current_weight, predicted_returns, realized_returns, lmda, kappa, fee):
    target_w, trans_mult = optimize_weight(cov_matrix, current_weight, predicted_returns, lmda, kappa, fee)
    new_weight = (target_w * realized_returns) / (target_w @ realized_returns + 1e-12)
    wealth_mult = trans_mult * (target_w @ realized_returns)
    return target_w, wealth_mult, new_weight


# ============================================================
# 3. Expert Evaluation Logic (STICKY ROUTING)
# ============================================================

def compute_ewma_series(wealth_history, ewma_alpha=0.1, return_days=3, num_scores=7):
    if len(wealth_history) < return_days + 5:
        return np.full(num_scores, -np.inf)

    wealth_arr = np.array(wealth_history)
    n_day_rets = (wealth_arr[return_days:] / wealth_arr[:-return_days]) - 1

    if len(n_day_rets) == 0:
        return np.full(num_scores, -np.inf)

    ewma_series = []
    ewma = n_day_rets[0]
    ewma_series.append(ewma)

    for r in n_day_rets[1:]:
        ewma = ewma_alpha * r + (1 - ewma_alpha) * ewma
        ewma_series.append(ewma)

    if len(ewma_series) < num_scores:
        pad = [-np.inf] * (num_scores - len(ewma_series))
        return np.array(pad + ewma_series)

    return np.array(ewma_series[-num_scores:])


def select_next_expert(experts, current_idx, ewma_alpha=0.1, return_days=3, min_consecutive_days=7):
    curr_scores = compute_ewma_series(
        experts[current_idx]['wealth_history'],
        ewma_alpha,
        return_days,
        min_consecutive_days
    )

    if np.isinf(curr_scores).any():
        return current_idx

    valid_candidates = []

    for idx, expert in enumerate(experts):
        if idx == current_idx:
            continue

        exp_scores = compute_ewma_series(
            expert['wealth_history'],
            ewma_alpha,
            return_days,
            min_consecutive_days
        )

        if np.all(exp_scores > curr_scores):
            valid_candidates.append((idx, exp_scores[-1]))

    if not valid_candidates:
        return current_idx

    best_idx = max(valid_candidates, key=lambda x: x[1])[0]
    return best_idx


# ============================================================
# 4. Main Algorithm Framework
# ============================================================

def adaptive_dmd_robust_algo(price_df, start_date, total_days=600, window=30,
                             lmda_list=None, kappa_list=None, sigma_list=None, fee=0.002,
                             burn_in_days=100,
                             ewma_alpha=0.1, return_days=3, min_consecutive_days=7,
                             verbose=True):
    param_grid = list(itertools.product(lmda_list, kappa_list, sigma_list))
    n_assets = price_df.shape[1]

    ret_df = 1.0 + price_df.pct_change().dropna(how='any')
    trading_days = ret_df.loc[pd.Timestamp(start_date):].index[:total_days + 1]

    if len(trading_days) < burn_in_days + 10:
        return None, np.nan

    experts = []
    for i, (lmda, kappa, sigma) in enumerate(param_grid):
        experts.append({
            'id': i, 'lmda': lmda, 'kappa': kappa, 'sigma': sigma,
            'wealth': 1.0,
            'current_weight': np.full(n_assets, 1.0 / n_assets),
            'wealth_history': [1.0],
            'param_str': f"λ={lmda:.4f}_κ={kappa:.1f}_σ={sigma:.1f}",
            'A': None, 'P': None,
            'last_pred_returns': None
        })

    def burnin_score(ex):
        h = ex['wealth_history']
        if len(h) < 3:
            return (-np.inf, -np.inf)
        rets = np.diff(h) / np.array(h[:-1])
        return (np.mean(rets), h[-1])

    tracking = None
    records = []

    for t in range(len(trading_days) - 1):
        curr_date = trading_days[t]
        next_date = trading_days[t + 1]

        hist_returns = ret_df.loc[:curr_date]
        if len(hist_returns) < window + 1:
            continue

        data_x = hist_returns.iloc[-window:].to_numpy().T
        data_y = hist_returns.iloc[-window - 1:-1].to_numpy().T
        x = hist_returns.iloc[-1].to_numpy().T
        y = hist_returns.iloc[-2].to_numpy().T

        cov_matrix = hist_returns.iloc[-window:].cov().values
        realized_returns = ret_df.loc[next_date].values

        for ex in experts:
            if ex['A'] is None:
                ex['A'], ex['P'] = get_parameter_A_P_gma(data_x, data_y)
            else:
                t_step = window + t
                ex['A'], ex['P'] = update_parameter_A_P_gma(x, y, ex['A'], ex['P'], ex['sigma'], t_step)

            raw_pred = ex['A'] @ x
            ex['last_pred_returns'] = np.nan_to_num(raw_pred, nan=1.0, posinf=2.0, neginf=0.0)

            _, mult, nw = update_portfolio(cov_matrix, ex['current_weight'], ex['last_pred_returns'],
                                           realized_returns, ex['lmda'], ex['kappa'], fee)
            ex['wealth'] *= mult
            ex['current_weight'] = nw
            ex['wealth_history'].append(ex['wealth'])

        if t < burn_in_days - 1:
            pass

        elif t == burn_in_days - 1:
            init_idx = max(range(len(experts)), key=lambda i: burnin_score(experts[i]))
            init_ex = experts[init_idx]

            tracking = {
                'wealth': init_ex['wealth'],
                'current_weight': init_ex['current_weight'].copy(),
                'wealth_history': init_ex['wealth_history'].copy(),
                'current_lmda': init_ex['lmda'],
                'current_kappa': init_ex['kappa'],
                'current_sigma': init_ex['sigma'],
                'current_idx': init_idx
            }

        else:
            new_idx = select_next_expert(experts, tracking['current_idx'], ewma_alpha, return_days,
                                         min_consecutive_days)
            if new_idx != tracking['current_idx']:
                new_ex = experts[new_idx]
                tracking['current_lmda'] = new_ex['lmda']
                tracking['current_kappa'] = new_ex['kappa']
                tracking['current_sigma'] = new_ex['sigma']
                tracking['current_idx'] = new_idx

            active_expert_pred = experts[tracking['current_idx']]['last_pred_returns']
            _, mult, nw = update_portfolio(cov_matrix, tracking['current_weight'], active_expert_pred,
                                           realized_returns, tracking['current_lmda'], tracking['current_kappa'], fee)

            tracking['wealth'] *= mult
            tracking['current_weight'] = nw
            tracking['wealth_history'].append(tracking['wealth'])

            records.append({
                'date': next_date,
                'wealth': tracking['wealth'],
                'lambda': round(tracking['current_lmda'], 5),
                'kappa': round(tracking['current_kappa'], 3),
                'sigma': round(tracking['current_sigma'], 3)
            })

    if tracking is None:
        return None, np.nan

    df = pd.DataFrame(records)
    df.set_index('date', inplace=True)
    return df, tracking['wealth']


# ============================================================
# Execution Block (MULTIPROCESSING)
# ============================================================
if __name__ == "__main__":
    tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'PG', 'KO', 'PEP', 'XOM', 'CVX',
        'JPM', 'BAC', 'WFC', 'GS', 'HD', 'MCD', 'NKE', 'DIS', 'VZ', 'T',
        'PFE', 'MRK', 'ABBV', 'LLY', 'UNH', 'CVS', 'WMT', 'COST', 'PM', 'MO',
        'IBM', 'ORCL', 'CSCO', 'INTC', 'QCOM', 'TXN', 'AVGO', 'CAT', 'DE',
        'BA', 'HON', 'GE', 'MMM', 'DOW', 'EMN', 'APD', 'NEE', 'DUK', 'SO'
    ]

    print("Downloading historical data...")
    price_df = yf.download(tickers, start='2018-01-01', end='2026-07-01',
                           progress=False, auto_adjust=True)['Close']
    price_df.index = price_df.index.tz_localize(None)
    price_df = price_df.dropna(how='any')

    # 1. Inner Grid Parameters (Passed to every process)
    fee = 0.002
    base_lmda_list = [fee * 10]
    base_kappa_list = [-5.0, -1.0, 0.0, 1.0, 5.0, 10.0]
    base_sigma_list = [0.0, 0.1, 0.5, 2.0, 5.0, 10.0]

    # 2. Outer Looping Grids (Distributed across CPU cores)
    window_grid = [20, 60, 120]
    alpha_grid = [0.05, 0.1, 0.2, 0.4]

    print(f"Distributing {len(window_grid) * len(alpha_grid)} macro configurations across CPU cores...")

    experts_list = []

    # Utilize all available CPU cores using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                adaptive_dmd_robust_algo,
                price_df,  # price_df
                '2021-02-21',  # start_date
                400,  # total_days
                wnd,  # window
                base_lmda_list,  # lmda_list
                base_kappa_list,  # kappa_list
                base_sigma_list,  # sigma_list
                fee,  # fee
                80,  # burn_in_days
                alpha_val,  # ewma_alpha
                3,  # return_days
                7,  # min_consecutive_days
                False  # verbose (disabled for multiprocessing)
            ): (wnd, alpha_val)
            for wnd in window_grid
            for alpha_val in alpha_grid
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            wnd, alpha_val = futures[future]
            try:
                result_df, final_wealth = future.result()
                if pd.notna(final_wealth):
                    experts_list.append({
                        'alpha': alpha_val,
                        'window': wnd,
                        'final_wealth': final_wealth
                    })
                    print(f"Completed: Window {wnd:3d} | Alpha {alpha_val:.2f} | Wealth: {final_wealth:.4f}")
            except Exception as e:
                print(f"Error executing Window {wnd}, Alpha {alpha_val}: {e}")

    # Output the final sorted leaderboard
    expert_df = pd.DataFrame(experts_list)
    if not expert_df.empty:
        print("\n\n" + "=" * 50)
        print("MACRO PARAMETER GRID RESULTS (SORTED)")
        print("=" * 50)
        print(expert_df.sort_values('final_wealth', ascending=False).to_string(index=False))
    else:
        print("All combinations resulted in NaN/Overflow.")
