'''
Li et al. proposed framework summary
1. Initialization (after collecting first m0 observations)
   - Build matrices from price relatives:
     X = price_relatives[:m0-1].T          # shape (m, m0-1)
     Y = price_relatives[1:m0].T           # shape (m, m0-1)

   - Compute initial matrices:
     Q = Y @ X.T
     P = inv(X @ X.T)
     A = Q @ P

   - Set current values:
     A_current = A
     P_current = P
     b = ones(m) / m                       # uniform portfolio
     t = m0

2. Main Online Loop (for each new period t = m0 to n-1)

   a. DMD Prediction
      v = A_current @ x_t                  # predicted vector for next period

   b. Portfolio Update (vectorized form of Equation 19)
      s = b @ v
      e = exp(eta * v / s)
      numer = b * e                        # element-wise product
      b = numer / numer.sum()

   c. Observe new price relative
      x_new = actual price relative vector at t+1
      (Optional) Update wealth: W = W * (b @ x_new)

   d. Recursive DMD Update (Theorem 1)
      - Compute gain:
        gamma = 1 / (1 + x_t.T @ P_current @ x_t)

      - Update auxiliary matrix P:
        log_factor = log(t + 1) / log(t)
        weight = log_factor ** sigma
        P_new = weight * (P_current - gamma * P_current @ x_t @ x_t.T @ P_current)

      - Update operator A:
        A_new = A_current + gamma * (x_new - A_current @ x_t) @ x_t.T @ P_current

      - Prepare for next iteration:
        A_current = A_new
        P_current = P_new
        x_t = x_new
        t = t + 1

3. End of loop
   - Final portfolio sequence and cumulative wealth are obtained.
'''

import yfinance as yf
import pandas as pd
import numpy as np


# 0. Data fetching
def get_price(price_df, decision_date, window=30):
  ret_df = 1 + price_df.pct_change().dropna(how='any')
  data_x = ret_df.iloc[-30:]
  data_y = ret_df.iloc[-31:-1]
  x = ret_df.iloc[-1]
  y = ret_df.iloc[-2]
  return data_x.to_numpy().T, data_y.to_numpy().T, x.to_numpy().T, y.to_numpy().T


# 1. Initialization (i=0)
def get_parameter_A_P_gma(data_x, data_y, y):
    Q = data_x @ data_y.T
    P_prev = np.linalg.inv(data_y @ data_y.T)
    P = np.linalg.inv(data_x @ data_x.T)
    A = Q @ P
    gma = 1 / (1 + y.T @ P_prev @ y)
    return A, P, gma


# 2. Portfolio Update
def get_weight(current_weight, A, x, eta=0.05):
    Ax = A @ x
    utility = Ax / (current_weight @ Ax)
    num = current_weight * np.exp(eta * utility)
    det = np.sum(current_weight * np.exp(eta * utility))
    target_weight =  num/det
    return target_weight


# 3. Update Cumulative Wealth
def cumulative_wealth(position, target_weight, x):
    return position * (target_weight @ x)


# 4. Update DMD Operator Recursively (i>0)
def update_parameter_A_P_gma(x, y, A, P, gma, sig, i):
    t = i + 2
    gma_prev = gma
    gma = 1 / (1 + y.T @ P @ y)
    x_hat_diff = x - A @ y
    A = A + gma * np.outer(x_hat_diff, y @ P)
    P =  (np.emath.logn(t, t + 1) ** sig) * (P - gma_prev * np.outer(P @ y, y @ P))
    return A, P, gma


# Main simulation
def run_li_sim(price_df, decision_date, total_days=60, window=30, eta=0.05, sig=500):
    position = 1.0
    n_assets = price_df.shape[1]
    current_weight = np.full(n_assets, 1.0 / n_assets)
    wealth_history = [1.0]
    decision_date = pd.Timestamp(decision_date)

    for i in range(total_days):
        if i == 0:
            data_x, data_y, x, y = get_price(price_df, decision_date, window)
            A, P, gma = get_parameter_A_P_gma(data_x, data_y, y)
            target_weight = get_weight(current_weight, A, x, eta)
            position = cumulative_wealth(position, target_weight, x)
            current_weight = target_weight
            wealth_history.append(position)
            decision_date = decision_date + pd.Timedelta(days=1)
        else:
            _, _, x, y = get_price(price_df, decision_date, window)
            A, P, gma = update_parameter_A_P_gma(x, y, A, P, gma, sig, i)
            target_weight = get_weight(current_weight, A, x, eta)
            position = cumulative_wealth(position, target_weight, x)
            current_weight = target_weight
            wealth_history.append(position)
            decision_date = decision_date + pd.Timedelta(days=1)

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

    eta_values = np.logspace(-3, 0.7, 9)
    sigma_values = [0, 0.1, 0.5, 1, 5]
    experts_list = []

    for eta in eta_values:
        for sigma in sigma_values:
            final_wealth = run_li_sim(
                price_df=price_df,
                decision_date='2024-01-01',
                total_days=60,
                window=30,
                eta=eta,
                sig=sigma
            )
            experts_list.append({
                'eta': round(eta, 5),
                'sigma': round(sigma, 3),
                'final_wealth': final_wealth,
                'name': f"l{eta:.4f}_k{sigma:.2f}"
            })

    expert_df = pd.DataFrame(experts_list)
    print(expert_df.sort_values('final_wealth', ascending=False).to_string(index=False))
