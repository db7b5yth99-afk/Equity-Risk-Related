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


# 0. Data fetching
def get_price(price_df, decision_date, window=30):
   if not isinstance(decision_date, pd.Timestamp):
       decision_date = pd.Timestamp(decision_date)
   hist_start = decision_date - pd.Timedelta(days=window + 10)
   if hist_start < price_df.index[0] or decision_date > price_df.index[-1] + pd.Timedelta(days=1):
       return None
   try:
       data = price_df.loc[hist_start:decision_date].copy()
       data = data.dropna(how='any')
       if data.empty or len(data) < 2:
           return None
       return data
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
   last_r = data.pct_change().iloc[-1].values
   return 1.0 + last_r

# 7-8. Wealth and weight update
def cumulative_wealth(position, transaction_loss, target_weight, realized_returns):
   return position * transaction_loss * (target_weight @ realized_returns)

def update_weight(old_target_weight, realized_returns):
    denom = old_target_weight @ realized_returns + 1e-12
    return (old_target_weight * realized_returns) / denom

# Main simulation for a SINGLE fixed (λ, κ) expert
def run_tsang_sim(price_df, start_date, total_days=60, window=30,
                 lmda=0.01, kappa=1.0, fee_proportion=0.001):
   position = 1.0
   n_assets = price_df.shape[1]
   current_weight = np.full(n_assets, 1.0 / n_assets)
   wealth_history = [1.0]
   start = pd.Timestamp(start_date)

   for t in range(total_days):
       data = get_price(price_df, start, window)
       if data is None or len(data) < max(5, window // 3):
           wealth_history.append(position)
           start += pd.Timedelta(days=1)
           continue

       predicted_returns = get_prediction(data, n_assets)
       cov_matrix = get_cov(data)

       target_weight = optimize_weight(
           cov_matrix, current_weight, predicted_returns,
           lmda, kappa, fee_proportion
       )

       w = find_transaction_loss(current_weight, target_weight, fee_proportion)

       next_decision = start + pd.Timedelta(days=1)
       next_data = get_price(price_df, next_decision, window=5)
       realized_r = updating_realized_return(next_data, n_assets)

       position = cumulative_wealth(position, w, target_weight, realized_r)
       current_weight = update_weight(target_weight, realized_r)
       wealth_history.append(position)
       start += pd.Timedelta(days=1)

   return wealth_history[-1]

# Create and evaluate experts
if __name__ == "__main__":
   tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
   price_df = yf.download(
       tickers,
       start='2023-01-01',
       end=pd.Timestamp.now() + pd.Timedelta(days=10),
       progress=False,
       auto_adjust=True
   )['Close'].dropna(how='any')

   lamda_values = np.logspace(np.log10(0.001), np.log10(0.1), 5)
   kappa_values = np.logspace(np.log10(0.1), np.log10(5), 4)

   experts_list = []
   for lamda in lamda_values:
       for kappa in kappa_values:
           final_wealth = run_tsang_sim(
               price_df=price_df,
               start_date='2024-01-01',
               total_days=60,
               window=30,
               lmda=lamda,
               kappa=kappa,
               fee_proportion=0.001
           )
           experts_list.append({
               'lambda': round(lamda, 5),
               'kappa': round(kappa, 3),
               'final_wealth': final_wealth,
               'name': f"l{lamda:.4f}_k{kappa:.2f}"
           })

   expert_df = pd.DataFrame(experts_list)
   print(expert_df.sort_values('final_wealth', ascending=False).to_string(index=False))
