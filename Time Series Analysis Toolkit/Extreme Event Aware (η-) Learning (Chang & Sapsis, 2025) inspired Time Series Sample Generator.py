import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import genpareto, norm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set random seeds for stable, reproducible initialization
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Historical Baseline & GPD Fitting
# ==========================================
print("Downloading baseline data...")
df = yf.download("^GSPC", start="2007-01-01", end="2021-12-31")['Close']
historical_returns = df.pct_change().dropna().to_numpy()

# Fit the GPD to the historical worst 5% of days
tau = 0.05
threshold = np.quantile(historical_returns, tau)
extreme_left_tail = historical_returns[historical_returns <= threshold]

# Convert to positive exceedances for the genpareto library
exceedances = -(extreme_left_tail - threshold)
shape, loc, scale = genpareto.fit(exceedances)

print(f"GPD Fitted - Shape: {shape:.4f}, Loc: {loc:.4f}, Scale: {scale:.4f}")

# Convert to tensors for the loss function
t_shape = torch.tensor(shape, dtype=torch.float32)
t_loc = torch.tensor(loc, dtype=torch.float32)
t_scale = torch.tensor(scale, dtype=torch.float32)
t_threshold = torch.tensor(threshold, dtype=torch.float32)


# ==========================================
# 2. The Simple Generative $\eta$-Map
# ==========================================
class EtaMapGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple, interpretable mapping network (1D to 1D)
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


model = EtaMapGenerator()


# ==========================================
# 3. The Generative $\eta$-Loss Function
# ==========================================
def gpd_icdf(q, loc, scale, shape):
    """Analytical Inverse CDF for the Generalized Pareto Distribution."""
    if shape != 0:
        return loc + (scale / shape) * ((1 - q) ** (-shape) - 1)
    else:
        return loc - scale * torch.log(1 - q)


def generative_eta_loss(synthetic_returns, naive_inputs, lambda_param):
    # 1. MSE: Keep the bulk of the data looking like normal market behavior
    mse = torch.nn.functional.mse_loss(synthetic_returns, naive_inputs)

    # 2. Extract the extreme left tail (worst 5% of synthetic returns)
    tail_threshold_val = torch.quantile(synthetic_returns, tau)
    tail_preds = synthetic_returns[synthetic_returns <= tail_threshold_val]

    if len(tail_preds) < 2:
        return mse, mse, torch.tensor(0.0)

    # Sorted from most negative (extreme crash) to least negative (threshold)
    tail_preds_sorted, _ = torch.sort(tail_preds.squeeze())

    # 3. Generate the theoretical GPD curve to match against
    # FIX 1: The GPD models the conditional tail. q must span near 1.0 down to 0.0.
    # FIX 2: Since tail_preds_sorted is ascending, q_grid must be descending.
    # The most negative prediction (index 0) pairs with the largest q (0.9999 = max crash).
    q_grid = torch.linspace(0.9999, 0.0001, len(tail_preds_sorted), device=synthetic_returns.device)

    theoretical_exceedances = gpd_icdf(q_grid, t_loc, t_scale, t_shape)

    # Map exceedances back to negative market returns
    tail_true_theoretical = t_threshold - theoretical_exceedances

    # 4. W1 Penalty: Absolute distance between synthetic tail and GPD curve
    W1 = torch.mean(torch.abs(tail_preds_sorted - tail_true_theoretical))

    total_loss = mse + (lambda_param * W1)
    return total_loss, mse, W1


# ==========================================
# 4. Training (Generator Calibration)
# ==========================================
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
epochs = 1000
batch_size = 2000

# Get mean and std of historical returns to generate naive base data
hist_mean = np.mean(historical_returns)
hist_std = np.std(historical_returns)

for epoch in range(epochs):
    # Generate naive random market days (Gaussian)
    naive_inputs = torch.normal(mean=hist_mean, std=hist_std, size=(batch_size, 1))

    # Pass through the $\eta$-map to get corrected synthetic returns
    synthetic_returns = model(naive_inputs)

    # Apply IICT (Inference-Informed Continual Training) concept:
    # Start pure MSE, then activate $\lambda$ to shape the tail
    current_lambda = 0.0 if epoch < 300 else 0.5

    loss, mse, W1 = generative_eta_loss(synthetic_returns, naive_inputs, current_lambda)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:04d} | Total Loss: {loss.item():.6f} | MSE: {mse.item():.6f} | W1: {W1.item():.6f} | Lambda: {current_lambda}")

# ==========================================
# 5. Generate Synthetic Stress-Test Paths
# ==========================================
model.eval()
with torch.no_grad():
    # Generate 10 different 1-year paths (252 trading days)
    # Shape: (252 days, 10 scenarios)
    naive_test_data = torch.normal(mean=hist_mean, std=hist_std, size=(252, 10))

    # Reshape to a flat list to pass through the neural network, then reshape back
    flat_naive = naive_test_data.view(-1, 1)
    flat_synthetic = model(flat_naive)

    # Reshape back to 252 days x 100 scenarios
    synthetic_returns_2d = flat_synthetic.view(252, 10).numpy()

# Calculate price paths starting at $100
starting_price = 2500.0

# Add 1 to returns, then cumprod along the time axis (axis=0)
price_paths = starting_price * np.cumprod(1 + synthetic_returns_2d, axis=0)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(price_paths, alpha=0.3, linewidth=1)
plt.title("100 Synthetic 1-Year Stock Paths ($\eta$-Mapped)")
plt.xlabel("Trading Days")
plt.ylabel("Simulated Price ($)")
plt.grid(alpha=0.3)
plt.show()
