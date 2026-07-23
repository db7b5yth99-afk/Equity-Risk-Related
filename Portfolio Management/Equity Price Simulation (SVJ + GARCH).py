import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Data
ticker = "MNTS"
time_horizon = "4y"
data = yf.download(ticker, period=time_horizon, interval="1d")['Close'].astype(float).dropna().squeeze()
returns = np.log(data / data.shift(1)).dropna()

# Parameter Calculation
moving_mu = returns.rolling(21).mean()
mu = moving_mu.iloc[-1]         # smoothen daily mu
gmodel = arch_model(returns, p=1, q=1)
gmodel_fit = gmodel.fit(disp="off")
moving_std = gmodel_fit.conditional_volatility
vt = moving_std.iloc[-1]
omega = gmodel_fit.params['omega']
alpha = gmodel_fit.params['alpha[1]']
beta = gmodel_fit.params['beta[1]']
kap = -np.log(alpha + beta)
theta = np.sqrt(omega / (1 - alpha - beta))
vov = np.sqrt(2 * kap * np.var(moving_std))
rho = np.corrcoef(moving_std.diff().iloc[-800:], returns.iloc[-800:])[0, 1]
abnr = 0.025                # hard-coded excess return

if abs(mu) > abs(returns[:-22].mean()) + abnr:
    print(f'Alert: Abnormal recent return (mu = {mu:.4f}), use long term ({time_horizon}-mu={returns.mean():.4f}) mu instead? (Y/N/set_0)')
    while True:
        uinput=input().lower()
        if uinput == 'y':
            mu = returns.mean()
            print(f'mu is set to {mu:.4f}')
            break
        elif uinput == 'set_0':
            mu = 0
            print(f'mu is set to {mu:.4f}')
            break
        elif uinput == 'n':
            print(f'mu remains {mu:.4f}')
            break
        else:
            print(f'Invalid input')

# Jump detection
jump_return = returns[(returns > (returns.mean() + theta * 3)) | (returns < (returns.mean() - theta * 3))]
lda = len(jump_return) / (len(returns))     # freq of daily jump return
mu_j = jump_return.mean()                   # jump mean
std_j = jump_return.std()                   # jump std
k = np.exp(mu_j + 0.5 * std_j**2) - 1

# Simulation
total_trials = 20000
quarter_pred = np.zeros(total_trials)
forcesell = 0
fsp = 1                     # force selling price

for j in range(total_trials):
    trials = 63
    dt = 1.0
    S = data.iloc[-1]
    vol = vt

    for i in range(trials):
        # Updating Parameters
        dWt1 = np.random.normal(0, 1.0)
        dWt2 = rho * dWt1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1.0)
        dvol = kap * (theta - vol) * dt + vov * dWt2
        diffusion = vol * dWt1
        vol = max(0, vol + dvol)          # prevent negative vol
        dNt = np.random.poisson(lda * dt)
        Z = np.random.normal(0, 1)
        jt = np.exp(mu_j + Z * std_j) if dNt > 0 else 1.0

        # The Three Puzzles
        drift = (mu - lda * k) * dt
        jump_effect = (jt - 1.0) * dNt

        # Result
        S = S * (1 + drift + diffusion + jump_effect)
        S = max(S, fsp)                      # high risk force sell
        if S == fsp:
            forcesell += 1
            S = 0
            break

    quarter_pred[j] = S

# Results
print(f"Current price: ${data.iloc[-1]:.4f}")
print(f"Total force sell (close price <1$) trial out of {total_trials} attempts: {forcesell}({(forcesell/total_trials)*100:.2f}%)")
print(f"Mean price after 1 quarter:   ${np.mean(quarter_pred):.4f}")
print(f"Median price after 1 quarter: ${np.median(quarter_pred):.4f}")
print(f"5% quantile (downside):       ${np.percentile(quarter_pred, 5):.4f}")
print(f"95% quantile (upside):        ${np.percentile(quarter_pred, 95):.4f}")
print(f"Prob of loss (>0% down):      {(quarter_pred < data.iloc[-1]).mean()*100:.1f}%")
sns.boxplot(y=quarter_pred[(quarter_pred>np.percentile(quarter_pred, 5))&(quarter_pred<np.percentile(quarter_pred, 95))])
plt.title("Price after 1 Quarter")
plt.show()