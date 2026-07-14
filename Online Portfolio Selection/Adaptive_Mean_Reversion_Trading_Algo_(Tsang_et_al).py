import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import itertools
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# Core functions
# ============================================================

def get_prediction(data):
    mean_prices = data.mean(axis=0).values
    current_prices = data.iloc[-1].values
    return mean_prices / current_prices


def get_cov(data):
    cov = data.pct_change().dropna().cov().values
    if cov.shape[0] == 0:
        return np.eye(data.shape[1]) * 0.0001
    return cov


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

    constraints = [{'type': 'ineq', 'fun': constraint}]
    bounds = [(0.0, 1.0)] * n

    # Avoid getting stuck if initial weight is all zeros (e.g. at portfolio start)
    initial_guess = current_weight * 0.99
    if np.sum(initial_guess) <= 0.01:
        initial_guess = np.ones(n) / n

    result = minimize(fun=objective, x0=initial_guess, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-4, 'maxiter': 50})
    if result.success:
        raw_b = np.maximum(result.x, 0.0)
        w = np.sum(raw_b)
        target_weight = raw_b / (w + 1e-12)
        return target_weight, w
    else:
        return current_weight.copy(), 1.0


def simulate_one_day(price_df, current_date, next_date, current_weight, lmda, kappa, fee, window):
    n_assets = price_df.shape[1]

    # Get index locations
    idx_curr = price_df.index.get_loc(current_date)
    start_idx = max(0, idx_curr - window + 1)

    # Extract trailing data window
    data_window = price_df.iloc[start_idx: idx_curr + 1]

    # DYNAMIC UNIVERSE: Only select assets that have NO NaNs in the trailing window
    valid_mask = data_window.notna().all(axis=0).values

    if valid_mask.sum() < 2:
        # Not enough assets have data yet; do not trade
        target_w = current_weight.copy()
        trans_mult = 1.0
    else:
        # Filter for active universe
        sub_data = data_window.loc[:, valid_mask]
        sub_current_weight = current_weight[valid_mask]

        predicted = get_prediction(sub_data)
        cov = get_cov(sub_data)

        # Optimize weights ONLY for active assets
        sub_target_w, trans_mult = optimize_weight(cov, sub_current_weight, predicted, lmda, kappa, fee)

        # Reconstruct full-size weight array (unlisted assets remain 0.0)
        target_w = np.zeros(n_assets)
        target_w[valid_mask] = sub_target_w

    # Realized returns from current to next date
    idx_next = price_df.index.get_loc(next_date)
    p_curr = price_df.iloc[idx_curr].values
    p_next = price_df.iloc[idx_next].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        realized = p_next / p_curr

    # If a stock is completely unlisted, division by NaN/Zero yields NaN. Convert these to 1.0 multiplier.
    realized = np.nan_to_num(realized, nan=1.0, posinf=1.0, neginf=1.0)

    wealth_mult = trans_mult * np.dot(target_w, realized)

    denom = np.dot(target_w, realized)
    if denom == 0:
        new_weight = target_w.copy()
    else:
        new_weight = (target_w * realized) / denom

    return target_w, wealth_mult, new_weight


# ============================================================
# Proper EWMA using alpha
# ============================================================

def compute_ewma_score(wealth_history, ewma_alpha=0.1, return_days=3):
    if len(wealth_history) < return_days + 5:
        return -np.inf

    wealth_arr = np.array(wealth_history)
    n_day_rets = (wealth_arr[return_days:] / wealth_arr[:-return_days]) - 1

    if len(n_day_rets) == 0:
        return -np.inf

    ewma = n_day_rets[0]
    for r in n_day_rets[1:]:
        ewma = ewma_alpha * r + (1 - ewma_alpha) * ewma

    return ewma


def select_next_expert(experts, current_idx, ewma_alpha=0.1, return_days=3):
    current_score = compute_ewma_score(experts[current_idx]['wealth_history'], ewma_alpha, return_days)
    best_idx = current_idx
    best_score = current_score

    for idx, expert in enumerate(experts):
        if idx == current_idx:
            continue
        score = compute_ewma_score(expert['wealth_history'], ewma_alpha, return_days)
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx if best_idx != current_idx else current_idx


# ============================================================
# Main Function with Back-Shifted Burn-in
# ============================================================

def adaptive_tsang_paper(price_df, start_date, total_days=600, window=51,
                         lmda_list=None, kappa_list=None, fee=0.002,
                         burn_in_days=100,
                         ewma_alpha=0.1, return_days=3, verbose=True):
    if lmda_list is None:
        lmda_list = [fee * (10 ** i) for i in [0, 1]]
    if kappa_list is None:
        negative = [-10.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.1]
        positive = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0]
        kappa_list = negative + positive

    param_grid = list(itertools.product(lmda_list, kappa_list))
    n_assets = price_df.shape[1]

    # Locate timeline and shift back for burn-in
    all_dates = price_df.index
    start_ts = pd.Timestamp(start_date)

    if start_ts not in all_dates:
        start_idx = all_dates.searchsorted(start_ts)
    else:
        start_idx = all_dates.get_loc(start_ts)

    if start_idx < burn_in_days:
        print(f"\n[ERROR] Not enough historical data before {start_date} for {burn_in_days} days of burn-in.")
        return None, None

    end_idx = min(start_idx + total_days + 1, len(all_dates))
    trading_days = all_dates[start_idx - burn_in_days: end_idx]
    actual_start_date = all_dates[start_idx].date()

    if len(trading_days) < burn_in_days + 10:
        print("\n[ERROR] Not enough future data to run the backtest.")
        return None, None

    # Calculate initial valid weights strictly for assets present on Day 1 of burn-in
    first_idx = all_dates.get_loc(trading_days[0])
    first_window_start = max(0, first_idx - window + 1)
    first_window_data = price_df.iloc[first_window_start: first_idx + 1]
    valid_first = first_window_data.notna().all(axis=0).values

    init_weight = np.zeros(n_assets)
    if valid_first.sum() > 0:
        init_weight[valid_first] = 1.0 / valid_first.sum()
    else:
        init_weight[0] = 1.0

    experts = []
    for i, (lmda, kappa) in enumerate(param_grid):
        experts.append({
            'id': i, 'lmda': lmda, 'kappa': kappa, 'wealth': 1.0,
            'current_weight': init_weight.copy(),
            'wealth_history': [1.0],
            'param_str': f"λ={lmda:.4f}_κ={kappa:.2f}"
        })

    # Burn-in
    print(f"Running {burn_in_days}-day burn-in leading up to {actual_start_date}...")
    for t in range(burn_in_days):
        curr, nxt = trading_days[t], trading_days[t + 1]
        for ex in experts:
            _, mult, nw = simulate_one_day(price_df, curr, nxt, ex['current_weight'],
                                           ex['lmda'], ex['kappa'], fee, window)
            ex['wealth'] *= mult
            ex['current_weight'] = nw
            ex['wealth_history'].append(ex['wealth'])

    def burnin_score(ex):
        h = ex['wealth_history']
        if len(h) < 3:
            return (-np.inf, -np.inf)
        rets = np.diff(h) / np.array(h[:-1])
        return (np.mean(rets), h[-1])

    init_idx = max(range(len(experts)), key=lambda i: burnin_score(experts[i]))
    init_ex = experts[init_idx]
    print(
        f"Initial expert selected on {actual_start_date}: {init_ex['param_str']} (avg return: {burnin_score(init_ex)[0]:.6f})")

    tracking = {
        'wealth': init_ex['wealth'],
        'current_weight': init_ex['current_weight'].copy(),
        'wealth_history': init_ex['wealth_history'].copy(),
        'current_lmda': init_ex['lmda'],
        'current_kappa': init_ex['kappa'],
        'current_idx': init_idx
    }

    records = []

    # Actual trading
    for t in range(burn_in_days, len(trading_days) - 1):
        curr, nxt = trading_days[t], trading_days[t + 1]

        for ex in experts:
            _, mult, nw = simulate_one_day(price_df, curr, nxt, ex['current_weight'],
                                           ex['lmda'], ex['kappa'], fee, window)
            ex['wealth'] *= mult
            ex['current_weight'] = nw
            ex['wealth_history'].append(ex['wealth'])

        new_idx = select_next_expert(experts, tracking['current_idx'],
                                     ewma_alpha=ewma_alpha,
                                     return_days=return_days)

        if new_idx != tracking['current_idx']:
            new_ex = experts[new_idx]
            tracking['current_lmda'] = new_ex['lmda']
            tracking['current_kappa'] = new_ex['kappa']
            tracking['current_idx'] = new_idx
            if verbose:
                print(f"  → Switched to {new_ex['param_str']} on {curr.date()}")

        _, mult, nw = simulate_one_day(price_df, curr, nxt, tracking['current_weight'],
                                       tracking['current_lmda'], tracking['current_kappa'],
                                       fee, window)
        tracking['wealth'] *= mult
        tracking['current_weight'] = nw
        tracking['wealth_history'].append(tracking['wealth'])

        records.append({
            'date': nxt,
            'wealth': tracking['wealth'],
            'lambda': round(tracking['current_lmda'], 5),
            'kappa': round(tracking['current_kappa'], 3)
        })

        if verbose and (t - burn_in_days + 1) % 50 == 0:
            print(f"Trade Day {t - burn_in_days + 1:4d} ({nxt.date()}) | Wealth: {tracking['wealth']:.4f} | "
                  f"λ={tracking['current_lmda']:.4f} | κ={tracking['current_kappa']:.2f}")

    print("\n" + "=" * 60)
    print(f"Final Wealth: {tracking['wealth']:.4f}")
    print(f"Final λ: {tracking['current_lmda']:.4f} | κ: {tracking['current_kappa']:.2f}")

    df = pd.DataFrame(records)
    df.set_index('date', inplace=True)
    return df, tracking['wealth']


# ============================================================
# Example
# ============================================================
if __name__ == "__main__":
    tickers = [
        'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AMZN', 'AVGO', 'BAC', 'BRK-B', 'CMCSA',
        'COST', 'CRM', 'CSCO', 'CVX', 'DHR', 'DIS', 'META', 'GOOG', 'GOOGL', 'HD',
        'INTC', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'MA', 'MCD', 'MDT', 'MRK',
        'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PYPL',
        'T', 'TMO', 'TSLA', 'TXN', 'UNH', 'V', 'VZ', 'WFC', 'WMT', 'XOM'
    ]

    # Set start date to 2015 to ensure sufficient historical data for the 80-day burn-in prior to 2016-01-01
    price_df = yf.download(tickers, start='2015-01-01', end='2026-07-01',
                           progress=False, auto_adjust=True)['Close']
    price_df.index = price_df.index.tz_localize(None)

    # Forward-fill allows stocks joining midway to stay inside the dataset
    price_df = price_df.ffill()

    for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for j in [None, [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0]]:
            print(f'\n\n{"=" * 40}\nParameter List\nAlpha : {i}\n{"=" * 40}')
            result_df, final_w = adaptive_tsang_paper(
                price_df=price_df,
                start_date='2016-01-01',
                total_days=1500,
                burn_in_days=80,
                kappa_list=j,
                ewma_alpha=i,
                return_days=1,
                verbose=True
            )

            # Robust Check: Only attempt to export if the backtest ran successfully
            if result_df is not None:
                if j is not None:
                    k = 'positive'
                else:
                    k = 'all'

                output_filename = f'portfolio_history_alpha_{i}_kappa_{k}_range.csv'
                result_df.to_csv(output_filename)
                print(f"\n[SUCCESS] Exported history to: {output_filename}")
            else:
                print(f"\n[WARNING] Backtest aborted. No CSV generated for Alpha: {i}, Kappa List: {j}")
