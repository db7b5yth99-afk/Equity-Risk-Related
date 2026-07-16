import yfinance as yf
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import genpareto
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

tickers = ["^GSPC"]
df=yf.download(tickers=tickers,  start="2010-01-01", end="2025-12-31")['Close'].squeeze()

"""Detect Extreme Event"""

df_ret = 1 + df.pct_change().dropna()
percentile_threshold = df_ret.quantile(0.01)
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

def generate_gpd_samples(num_samples, loc, scale, shape):
    gp_dist = torch.distributions.GeneralizedPareto(loc=loc, scale=scale, concentration=shape)
    samples = gp_dist.sample([num_samples])
    return samples

def custom_eta_loss(y_pred, y_true, lambda_param, loc, scale, shape):
    mse = torch.nn.functional.mse_loss(y_pred, y_true)
    tail_preds = y_pred[y_pred <= torch.quantile(y_pred, 0.001)]
    tail_true = generate_gpd_samples(len(tail_preds), loc, scale, shape)
    tail_preds_sorted, _ = torch.sort(tail_preds)
    tail_true_sorted, _ = torch.sort(tail_true)
    W1 = torch.mean(torch.abs(tail_preds_sorted - tail_true_sorted))
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
epochs = 80
lambda_param = 0.01

for epoch in range(epochs):
    y_pred = model(X_train)
    loss, mse, W1 = custom_eta_loss(y_pred, y_train, lambda_param, loc, scale, shape)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

"""Model Testing"""

model.eval()

with torch.no_grad():
    y_test_pred = model(X_test)
    test_loss, test_mse, test_W1 = custom_eta_loss(y_test_pred, y_test, lambda_param, loc, scale, shape)
    print(f"Training Complete! \nFinal Test Loss: {test_loss.item():.4f}\nFinal MSE: {test_mse.item():.4f}\nFinal W1: {test_W1.item():.4f}")





