'''
Tsang et al. proposed framework summary
for t = 1, …, n do
    1. Update parameters adaptively using expert performance:
       - For λ (SB scheme): Track cumulative wealth of λ-experts over recent window W.
         Switch to the expert l' if its moving average significantly outperforms current expert
         (exceeds ¯S + z ˆσ with indifference zone δ).
       - For κ (Top-K or SB scheme): Rank experts by cumulative wealth and update κ as
         average of top-K performing κ values (or switch via SB logic).

    2. Predict next-period price relative ˜x_t (e.g., moving average reversion predictor).

    3. Construct sample covariance Σ from recent m+1 observations (if available).

    4. Solve the robust optimization model for new portfolio weights b_t:
       maximize   bᵀ ˜x_t − λ ‖ˆb_{t−1} − b‖₁ − κ ‖U b‖₂
       subject to bᵀ 1 + γ ‖ˆb_{t−1} − b‖₁ ≤ 1,
                  b ≥ 0
       (SOCP formulation; U from Cholesky of Σ. Fall back to b_t = ˆb_{t−1} if insufficient data.)

    5. Compute net proportion w_{t−1} of rebalancing from ˆb_{t−1} by solving the transaction cost equation:
       w + γ ‖ˆb_{t−1} − w b_t‖₁ = 1.

    6. Observe the realized return x_t.

    7. Update cumulative wealth: S_t = S_{t−1} w_{t−1} (b_tᵀ x_t).

    8. Obtain current portfolio weights: ˆb_t = (b_t ⊙ x_t) / (b_tᵀ x_t).

    9. Update cumulative wealth records for all experts (using their individual b_t decisions).
end
'''

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import concurrent.futures
import itertools


# 0. Data fetching
def get_price(price_df, decision_date, window=5):
    """Simple version: return last `window` rows of raw prices up to decision_date."""
    if not isinstance(decision_date, pd.Timestamp):
        decision_date = pd.Timestamp(decision_date)
    try:
        data = price_df.loc[:decision_date].iloc[-window:].copy()
        data = data.dropna(how='any')
        return data if len(data) >= 2 else None
    except Exception:
        return None

# 2. Prediction
def get_prediction(data, n_assets):
    if data is None or len(data) < 2:
        return np.ones(n_assets)
    mean_r = data.pct_change().dropna().mean().values
    return 1.0 + mean_r

# 3. Covariance
def get_cov(data):
    if data is None or len(data) < 2:
        n = data.shape[1] if data is not None else 1
        return np.eye(n) * 0.0001
    return data.pct_change().dropna().cov().values

# 4. Portfolio optimization
def optimize_weight(cov_matrix, current_weight, predicted_returns, lmda, kappa, fee_proportion=0.001):
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
    initial_guess = np.full(n, 1.0 / n)
    result = minimize(
        fun=objective,
        x0=initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-8, 'maxiter': 200}
    )
    if result.success:
        weights = np.maximum(result.x, 0.0)
        weights /= (np.sum(weights) + 1e-12)
        return weights
    else:
        return current_weight.copy()

# 5. Transaction cost
def find_transaction_loss(current_weight, target_weight, fee_proportion=0.001):
    l1 = np.abs(current_weight - target_weight).sum()
    w = 1.0 / (1.0 + fee_proportion * l1)
    return max(w, 0.0)

# 6. Realized return
def updating_realized_return(data, n_assets):
    if data is None or len(data) < 2:
        return np.ones(n_assets)
    last_r = data.pct_change().dropna().iloc[-1].values
    return 1.0 + last_r


# 7-8. Wealth and weight update
def cumulative_wealth(position, transaction_loss, target_weight, realized_returns):
    return position * transaction_loss * (target_weight @ realized_returns)

def update_weight(old_target_weight, realized_returns):
    denom = old_target_weight @ realized_returns + 1e-12
    return (old_target_weight * realized_returns) / denom

# Main simulation for a SINGLE fixed (λ, κ) expert
def run_tsang_sim(price_df, decision_date, total_days=60, window=5,
                  lmda=0.01, kappa=1.0, fee_proportion=0.001):
    position = 1.0
    n_assets = price_df.shape[1]
    current_weight = np.full(n_assets, 1.0 / n_assets)
    target_weight = current_weight.copy()
    wealth_history = [1.0]

    decision_date = pd.Timestamp(decision_date)

    # Extract only actual trading days from the dataframe.
    # We grab total_days + 1 so we always have a "next_date" to calculate realized returns.
    trading_days = price_df.loc[decision_date:].index[:total_days + 1]

    for i in range(len(trading_days) - 1):
        current_date = trading_days[i]
        next_date = trading_days[i + 1]

        data = get_price(price_df, current_date, window)

        if data is not None:
            predicted_returns = get_prediction(data, n_assets)
            cov_matrix = get_cov(data)
            target_weight = optimize_weight(
                cov_matrix=cov_matrix,
                current_weight=current_weight,
                predicted_returns=predicted_returns,
                lmda=lmda,
                kappa=kappa,
                fee_proportion=fee_proportion
            )
            w = find_transaction_loss(current_weight, target_weight, fee_proportion)
        else:
            w = 1.0  # keep previous target

        # Fetch realized return for the exact next trading day
        next_data = get_price(price_df, next_date, window=10)
        realized_r = updating_realized_return(next_data, n_assets)

        position = cumulative_wealth(position, w, target_weight, realized_r)
        current_weight = update_weight(target_weight, realized_r)
        wealth_history.append(position)

    return wealth_history[-1]

# Create and evaluate experts
if __name__ == "__main__":
    tickers = [
        'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AMZN', 'AVGO', 'BAC', 'BRK-B', 'CMCSA',
        'COST', 'CRM', 'CSCO', 'CVX', 'DHR', 'DIS', 'META', 'GOOG', 'GOOGL', 'HD',  # <-- Changed FB to META
        'INTC', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'MA', 'MCD', 'MDT', 'MRK',
        'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PYPL',
        'T', 'TMO', 'TSLA', 'TXN', 'UNH', 'V', 'VZ', 'WFC', 'WMT', 'XOM'
    ]

    price_df = yf.download(
        tickers=tickers,
        start='2015-07-01',
        end='2026-07-01',
        progress=False,
        auto_adjust=True
    )['Close']

    # Strip timezones so it matches pd.Timestamp('YYYY-MM-DD')
    price_df.index = price_df.index.tz_localize(None)

    # Now it is safe to drop NA
    price_df = price_df.dropna(how='any')

    gamma = 0.002
    lamda_values = np.logspace(np.log10(gamma), np.log10(100 * gamma), num=10)
    kappa_values = np.logspace(np.log10(0.1), np.log10(10), num=8)
    experts_list = []

    # 1. Flatten the nested loops into a single list of combinations
    param_grid = list(itertools.product(lamda_values, kappa_values))

    # 2. Fire up the ProcessPoolExecutor to use all CPU cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks simultaneously
        futures = {
            executor.submit(
                run_tsang_sim,
                price_df,
                '2016-01-04',
                200,  # total_days
                5,  # window
                lamda,
                kappa,
                gamma
            ): (lamda, kappa) for lamda, kappa in param_grid
        }

        # 3. Harvest the results as soon as they finish processing
        for future in concurrent.futures.as_completed(futures):
            lamda, kappa = futures[future]
            try:
                final_wealth = future.result()
                experts_list.append({
                    'lambda': round(lamda, 5),
                    'kappa': round(kappa, 3),
                    'final_wealth': final_wealth,
                    'name': f"l{lamda:.4f}_k{kappa:.2f}"
                })
            except Exception as e:
                print(f"Expert l{lamda:.4f}_k{kappa:.2f} failed: {e}")

    expert_df = pd.DataFrame(experts_list)
    print(expert_df.sort_values('final_wealth', ascending=False).to_string(index=False))
    
'''
     lambda  kappa  final_wealth           name
    0.00928  0.100      1.370273  l0.0093_k0.10
    0.00928  0.193      1.343596  l0.0093_k0.19
    0.00928  0.373      1.291692  l0.0093_k0.37
    0.00928  0.720      1.282825  l0.0093_k0.72
    0.00928  1.389      1.203125  l0.0093_k1.39
    0.01549  0.720      1.195878  l0.0155_k0.72
    0.01549  1.389      1.192313  l0.0155_k1.39
    0.00557  0.100      1.188466  l0.0056_k0.10
    0.01549  2.683      1.172615  l0.0155_k2.68
    0.00557  0.193      1.160628  l0.0056_k0.19
    0.11990  0.720      1.131207  l0.1199_k0.72
    0.11990  0.373      1.131203  l0.1199_k0.37
    0.11990  2.683      1.131202  l0.1199_k2.68
    0.20000  0.193      1.131202  l0.2000_k0.19
    0.20000  2.683      1.131201  l0.2000_k2.68
    0.20000 10.000      1.131201 l0.2000_k10.00
    0.20000  1.389      1.131201  l0.2000_k1.39
    0.11990  0.100      1.131200  l0.1199_k0.10
    0.20000  0.100      1.131199  l0.2000_k0.10
    0.20000  5.179      1.131199  l0.2000_k5.18
    0.11990 10.000      1.131199 l0.1199_k10.00
    0.11990  5.179      1.131199  l0.1199_k5.18
    0.20000  0.720      1.131198  l0.2000_k0.72
    0.20000  0.373      1.131198  l0.2000_k0.37
    0.11990  1.389      1.131198  l0.1199_k1.39
    0.07188  1.389      1.131198  l0.0719_k1.39
    0.11990  0.193      1.131193  l0.1199_k0.19
    0.00928  2.683      1.129394  l0.0093_k2.68
    0.07188 10.000      1.128571 l0.0719_k10.00
    0.07188  5.179      1.124722  l0.0719_k5.18
    0.00557  1.389      1.123439  l0.0056_k1.39
    0.01549  5.179      1.122571  l0.0155_k5.18
    0.00928  5.179      1.117237  l0.0093_k5.18
    0.04309 10.000      1.114443 l0.0431_k10.00
    0.07188  2.683      1.114206  l0.0719_k2.68
    0.02583  5.179      1.112587  l0.0258_k5.18
    0.02583 10.000      1.107548 l0.0258_k10.00
    0.00928 10.000      1.098162 l0.0093_k10.00
    0.01549 10.000      1.096726 l0.0155_k10.00
    0.00557  2.683      1.095696  l0.0056_k2.68
    0.00334  2.683      1.090002  l0.0033_k2.68
    0.00557 10.000      1.089513 l0.0056_k10.00
    0.00557  5.179      1.088259  l0.0056_k5.18
    0.00557  0.720      1.080885  l0.0056_k0.72
    0.00334 10.000      1.078742 l0.0033_k10.00
    0.01549  0.373      1.076069  l0.0155_k0.37
    0.00200  2.683      1.072061  l0.0020_k2.68
    0.00200 10.000      1.066855 l0.0020_k10.00
    0.01549  0.193      1.064963  l0.0155_k0.19
    0.00557  0.373      1.062510  l0.0056_k0.37
    0.00334  5.179      1.057784  l0.0033_k5.18
    0.00334  0.100      1.055227  l0.0033_k0.10
    0.00334  0.193      1.050593  l0.0033_k0.19
    0.07188  0.720      1.036963  l0.0719_k0.72
    0.00334  0.373      1.035842  l0.0033_k0.37
    0.00200  5.179      1.034765  l0.0020_k5.18
    0.02583  0.193      1.031819  l0.0258_k0.19
    0.07188  0.373      1.029859  l0.0719_k0.37
    0.07188  0.193      1.029723  l0.0719_k0.19
    0.07188  0.100      1.029059  l0.0719_k0.10
    0.04309  2.683      1.028980  l0.0431_k2.68
    0.04309  0.373      1.028980  l0.0431_k0.37
    0.04309  0.100      1.028980  l0.0431_k0.10
    0.04309  0.720      1.028980  l0.0431_k0.72
    0.04309  1.389      1.028980  l0.0431_k1.39
    0.04309  0.193      1.028979  l0.0431_k0.19
    0.04309  5.179      1.028364  l0.0431_k5.18
    0.02583  2.683      1.028333  l0.0258_k2.68
    0.01549  0.100      1.021571  l0.0155_k0.10
    0.02583  0.373      1.019160  l0.0258_k0.37
    0.02583  0.100      1.017602  l0.0258_k0.10
    0.02583  0.720      1.016096  l0.0258_k0.72
    0.02583  1.389      1.012221  l0.0258_k1.39
    0.00334  1.389      1.001620  l0.0033_k1.39
    0.00200  1.389      0.999226  l0.0020_k1.39
    0.00200  0.720      0.980139  l0.0020_k0.72
    0.00334  0.720      0.978876  l0.0033_k0.72
    0.00200  0.373      0.959032  l0.0020_k0.37
    0.00200  0.193      0.913180  l0.0020_k0.19
    0.00200  0.100      0.886280  l0.0020_k0.10
'''
