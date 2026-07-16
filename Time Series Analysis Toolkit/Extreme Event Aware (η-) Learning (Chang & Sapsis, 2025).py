mport yfinance as yf
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import genpareto
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

tickers = ["^GSPC"]
df=yf.download(tickers=tickers,  start="2010-01-01", end="2025-12-31")['Close'].squeeze()

"""Seperate Adverse and Normal Market Condition"""

df_ret = 1 + df.pct_change().dropna()
percentile_threshold = df_ret.quantile(0.05)
ext_idx = df_ret[df_ret <= percentile_threshold].index
norm_idx = [i for i in df_ret.index if i not in set(ext_idx)]
norm_df = df_ret.loc[norm_idx]
ext_df = df_ret.loc[ext_idx]

"""Neuro-Network & Penalty Construction"""

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=32, batch_first=True)
        self.activation = nn.LeakyReLU()
        self.output_layer = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        x = out[:, -1, :]
        x = self.activation(x)
        x = self.output_layer(x)
        return x

def gpd_icdf(q, loc, scale, shape):
    """Analytical Inverse CDF for the Generalized Pareto Distribution."""
    if shape != 0:
        return loc + (scale / shape) * ((1 - q)**(-shape) - 1)
    else:
        return loc - scale * torch.log(1 - q)

def custom_eta_loss(y_pred, y_true, lambda_param, loc, scale, shape, threshold, tau=0.05):
    mse = torch.nn.functional.mse_loss(y_pred, y_true)
    tail_threshold_val = torch.quantile(y_pred, tau)
    tail_preds = y_pred[y_pred <= tail_threshold_val]
    if len(tail_preds) < 2:
        return mse, mse, torch.tensor(0.0)
    tail_preds_sorted, _ = torch.sort(tail_preds.squeeze())
    # Grid targeting the extreme left tail
    q_grid = torch.linspace(0.0001, tau, len(tail_preds_sorted), device=y_pred.device)
    # Calculate theoretical positive exceedances
    theoretical_exceedances = gpd_icdf(q_grid, loc, scale, shape)
    # Map back to the original negative return space
    tail_true_theoretical = threshold - theoretical_exceedances
    W1 = torch.mean(torch.abs(tail_preds_sorted - tail_true_theoretical))
    total_loss = mse + lambda_param * W1
    return total_loss, mse, W1

"""Fitting & Training"""

# Convert left tail to positive exceedances
exceedances = -(ext_df - percentile_threshold)
shape, loc, scale = genpareto.fit(exceedances.to_numpy())
shape, loc, scale = torch.tensor(shape), torch.tensor(loc), torch.tensor(scale)
all_data = df_ret.to_numpy()
X_raw_all = sliding_window_view(all_data[:-1], window_shape=5)
y_raw_all = all_data[5:]
split_idx = int(len(X_raw_all) * 0.8)

# Target mask: True for normal days, False for crashes
target_mask = y_raw_all > percentile_threshold

X_train_raw = X_raw_all[:split_idx]
y_train_raw = y_raw_all[:split_idx]
train_mask = target_mask[:split_idx]

# Model now sees pre-crash inputs, but normal targets
X_train = torch.tensor(X_train_raw[train_mask], dtype=torch.float32)
y_train = torch.tensor(y_train_raw[train_mask], dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_raw_all[split_idx:], dtype=torch.float32)
y_test = torch.tensor(y_raw_all[split_idx:], dtype=torch.float32).view(-1, 1)

# Optimizer
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 200
lambda_param = 0.8

for epoch in range(epochs):
    # Pre-training
    current_lambda = 0.0 if epoch < 75  else lambda_param

    y_pred = model(X_train)

    loss, mse, W1 = custom_eta_loss(
        y_pred,
        y_train,
        current_lambda,
        loc,
        scale,
        shape,
        percentile_threshold,
        tau=0.05
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | MSE: {mse.item():.4f} | W1: {W1.item():.4f}")

"""Model Testing"""

model.eval()

with torch.no_grad():
    y_test_pred = model(X_test)
    test_loss, test_mse, test_W1 = custom_eta_loss(y_test_pred, y_test, lambda_param, loc, scale, shape, percentile_threshold,)
    print(f"Training Complete! \nFinal Test Loss: {test_loss.item():.4f}\nFinal MSE: {test_mse.item():.4f}\nFinal W1: {test_W1.item():.4f}")import yfinance as yf
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import genpareto
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

tickers = ["^GSPC"]
df=yf.download(tickers=tickers,  start="2010-01-01", end="2025-12-31")['Close'].squeeze()

"""Seperate Adverse and Normal Market Condition"""

df_ret = 1 + df.pct_change().dropna()
percentile_threshold = df_ret.quantile(0.05)
ext_idx = df_ret[df_ret <= percentile_threshold].index
norm_idx = [i for i in df_ret.index if i not in set(ext_idx)]
norm_df = df_ret.loc[norm_idx]
ext_df = df_ret.loc[ext_idx]

"""Neuro-Network & Penalty Construction"""

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(in_features=5, out_features=16)
        self.activation = nn.LeakyReLU()
        self.output_layer = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

def gpd_icdf(q, loc, scale, shape):
    if shape != 0:
        return loc + (scale / shape) * ((1 - q)**(-shape) - 1)
    else:
        return loc - scale * torch.log(1 - q)

def custom_eta_loss(y_pred, y_true, lambda_param, loc, scale, shape, tau=0.05):
    mse = torch.nn.functional.mse_loss(y_pred, y_true)
    tail_threshold = torch.quantile(y_pred, tau)
    tail_preds = y_pred[y_pred <= tail_threshold]
    if len(tail_preds) < 2:
        return mse, mse, torch.tensor(0.0)
    tail_preds_sorted, _ = torch.sort(tail_preds.squeeze())
    q_grid = torch.linspace(0.0001, tau, len(tail_preds_sorted), device=y_pred.device)
    tail_true_theoretical = gpd_icdf(q_grid, loc, scale, shape)
    W1 = torch.mean(torch.abs(tail_preds_sorted - tail_true_theoretical))
    total_loss = mse + lambda_param * W1
    return total_loss, mse, W1

"""Fitting & Training"""

# Extreme Data
shape, loc, scale = genpareto.fit(ext_df.to_numpy())
loc = torch.tensor(loc)
scale = torch.tensor(scale)
shape = torch.tensor(shape)

# Normal Data
all_data = df_ret.to_numpy()
X_raw_all = sliding_window_view(all_data[:-1], window_shape=5)
y_raw_all = all_data[5:]
split_idx = int(len(X_raw_all) * 0.8)
X_test = torch.tensor(X_raw_all[split_idx:], dtype=torch.float32)
y_test = torch.tensor(y_raw_all[split_idx:], dtype=torch.float32).view(-1, 1)
masked_series = df_ret.copy()
masked_series.loc[ext_idx] = np.nan
masked_data = masked_series.to_numpy()
X_raw_masked = sliding_window_view(masked_data[:-1], window_shape=5)
y_raw_masked = masked_data[5:]
X_train_raw = X_raw_masked[:split_idx]
y_train_raw = y_raw_masked[:split_idx]
valid_mask = ~np.isnan(X_train_raw).any(axis=1) & ~np.isnan(y_train_raw)
X_train = torch.tensor(X_train_raw[valid_mask], dtype=torch.float32)
y_train = torch.tensor(y_train_raw[valid_mask], dtype=torch.float32).view(-1, 1)

# Optimizer
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 200 
lambda_param = 0.8

for epoch in range(epochs):
    # Pre-training
    current_lambda = 0.0 if epoch < 75  else lambda_param
    y_pred = model(X_train)
    loss, mse, W1 = custom_eta_loss(
        y_pred,
        y_train,
        current_lambda,
        loc,
        scale,
        shape,
        tau=0.05 
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | MSE: {mse.item():.4f} | W1: {W1.item():.4f}")

"""Model Testing"""

model.eval()

with torch.no_grad():
    y_test_pred = model(X_test)
    test_loss, test_mse, test_W1 = custom_eta_loss(y_test_pred, y_test, lambda_param, loc, scale, shape)
    print(f"Training Complete! \nFinal Test Loss: {test_loss.item():.4f}\nFinal MSE: {test_mse.item():.4f}\nFinal W1: {test_W1.item():.4f}")





