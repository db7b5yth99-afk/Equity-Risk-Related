import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import itertools
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# Core functions (kept clean from your original)
# ============================================================

def get_price(price_df, decision_date, window=51):
    if not isinstance(decision_date, pd.Timestamp):
        decision_date = pd.Timestamp(decision_date)
    try:
        data = price_df.loc[:decision_date].iloc[-window:].copy()
        data = data.dropna(how='any')
        return data if len(data) >= 2 else None
    except Exception:
        return None


def get_prediction(data, n_assets):
    if data is None or len(data) < 2:
        return np.ones(n_assets)
    mean_prices = data.mean(axis=0).values
    current_prices = data.iloc[-1].values
    return mean_prices / current_prices


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
        risk_penalty = kappa * np.linalg.norm(U @ b, ord=2)
        return -(port_return - l1_penalty - risk_penalty)

    def constraint(b):
        return 1.0 - (np.sum(b) + fee_proportion * np.linalg.norm(current_weight - b, ord=1))

    constraints = [{'type': 'ineq', 'fun': constraint}]
    bounds = [(0.0, 1.0)] * n
    initial_guess = current_weight * 0.99
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
    data = get_price(price_df, current_date, window)
    if data is not None:
        predicted = get_prediction(data, n_assets)
        cov = get_cov(data)
        target_w, trans_mult = optimize_weight(cov, current_weight, predicted, lmda, kappa, fee)
    else:
        target_w = current_weight.copy()
        trans_mult = 1.0
    next_data = get_price(price_df, next_date, window=10)
    realized = 1.0 + next_data.pct_change().dropna().iloc[-1].values if len(next_data) >= 2 else np.ones(n_assets)
    new_weight = (target_w * realized) / (target_w @ realized + 1e-12)
    wealth_mult = trans_mult * (target_w @ realized)
    return target_w, wealth_mult, new_weight


# ============================================================
# Proper EWMA using alpha (the 'a' in your formula)
# ============================================================

def compute_ewma_score(wealth_history, ewma_alpha=0.1, return_days=3):
    """
    EWMA score using the exact formula you showed:
        EWMA(t) = α * x(t) + (1 - α) * EWMA(t-1)

    ewma_alpha controls the decay:
        - Higher α → more weight on newest observation (more reactive)
        - Lower α → more weight on history (smoother)
    """
    if len(wealth_history) < return_days + 5:
        return -np.inf

    wealth_arr = np.array(wealth_history)
    n_day_rets = (wealth_arr[return_days:] / wealth_arr[:-return_days]) - 1

    if len(n_day_rets) == 0:
        return -np.inf

    # Recursive EWMA with given alpha
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
# Main Function with ewma_alpha as hyperparameter
# ============================================================

def adaptive_tsang_paper(price_df, start_date, total_days=600, window=51,
                         lmda_list=None, kappa_list=None, fee=0.002,
                         burn_in_days=100,
                         ewma_alpha=0.1, return_days=3, verbose=True):
    """
    Clean adaptive version.
    - ewma_alpha is the direct 'a' parameter from your formula.
    """
    if lmda_list is None:
        lmda_list = [fee * (10 ** i) for i in [0, 1]]
    if kappa_list is None:
        negative = [-10.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.1]
        positive = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0]
        kappa_list = negative + positive

    param_grid = list(itertools.product(lmda_list, kappa_list))
    n_assets = price_df.shape[1]
    trading_days = price_df.loc[pd.Timestamp(start_date):].index[:total_days + 1]

    if len(trading_days) < burn_in_days + 10:
        print("Not enough data.")
        return None

    experts = []
    for i, (lmda, kappa) in enumerate(param_grid):
        experts.append({
            'id': i, 'lmda': lmda, 'kappa': kappa, 'wealth': 1.0,
            'current_weight': np.full(n_assets, 1.0 / n_assets),
            'wealth_history': [1.0],
            'param_str': f"λ={lmda:.4f}_κ={kappa:.2f}"
        })

    # Burn-in
    print(f"Running {burn_in_days}-day burn-in...")
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
    print(f"Initial expert: {init_ex['param_str']} (avg return: {burnin_score(init_ex)[0]:.6f})")

    tracking = {
        'wealth': init_ex['wealth'],
        'current_weight': init_ex['current_weight'].copy(),
        'wealth_history': init_ex['wealth_history'].copy(),
        'current_lmda': init_ex['lmda'],
        'current_kappa': init_ex['kappa'],
        'current_idx': init_idx
    }

    records = []

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
                print(f"  → Switched to {new_ex['param_str']}")

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

        if verbose and (t + 1) % 50 == 0:
            print(f"Day {t + 1:4d} | Wealth: {tracking['wealth']:.4f} | "
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
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'PG', 'KO', 'PEP', 'XOM', 'CVX',
        'JPM', 'BAC', 'WFC', 'GS', 'HD', 'MCD', 'NKE', 'DIS', 'VZ', 'T',
        'PFE', 'MRK', 'ABBV', 'LLY', 'UNH', 'CVS', 'WMT', 'COST', 'PM', 'MO',
        'IBM', 'ORCL', 'CSCO', 'INTC', 'QCOM', 'TXN', 'AVGO', 'CAT', 'DE',
        'BA', 'HON', 'GE', 'MMM', 'DOW', 'EMN', 'APD', 'NEE', 'DUK', 'SO'
    ]
    price_df = yf.download(tickers, start='2018-01-01', end='2026-07-01',
                           progress=False, auto_adjust=True)['Close']
    price_df.index = price_df.index.tz_localize(None)
    price_df = price_df.dropna(how='any')
    for i in [0.41, 0.44, 0.47, 0.5, 0.53, 0.56, 0.59]:
        for j in [1,2,3]:
            print(f'\n\nParameter List\nAlpha : {i}\nReturn day Smoothing : {j}')
            result_df, final_w = adaptive_tsang_paper(
                price_df=price_df,
                start_date='2019-01-01',
                total_days= 1500,
                burn_in_days=80,
                ewma_alpha=i,
                return_days=j,
                verbose=True
            )
            print("\nLast 10 rows:")
            print(result_df.tail(10))
