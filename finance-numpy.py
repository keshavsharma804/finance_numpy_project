# -*- coding: utf-8 -*-
"""
Original file is located at
https://colab.research.google.com/drive/18fsJkwjdWDHTnbbwvjRX4AV7LTfTLoSc
"""

from readline import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

np.random.seed(42)

months = np.array(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
income = (np.full(12, 5000.0) + np.random.normal(0, 200, 12)).astype(np.float64)

food = np.round(np.random.normal(400, 40, 12), 2)
rent = np.round(np.full(12, 1800.0) + np.random.normal(0, 50, 12), 2)
travel = np.round(np.abs(np.random.normal(150, 120, 12)), 2)


expenses_df = pd.DataFrame({
    "month": months,
    "income": income,
    "food": food,
    "rent": rent,
    "travel": travel
})
display(expenses_df)

total_expenses = food + rent + travel
savings = income - total_expenses

high_rent_mask = rent > 0.5 * income
print("\nMonths where rent > 50% of income:", months[high_rent_mask].tolist())

print("\nIndexing & Slicing examples:")
print("First month income:", income[0])
print("Q1 months (Jan-Mar) total expenses:", np.sum(total_expenses[:3]))
print("Compare Q1 vs Q2 total expenses:", np.sum(total_expenses[:3]), "vs", np.sum(total_expenses[3:6]))

total_expenses = food + rent + travel
savings = income - total_expenses

print("\nIndexing & Slicing examples:")
print("First month income:", income[0])
print("Q1 months (Jan-Mar) total expenses:", np.sum(total_expenses[:3]))
print("Compare Q1 vs Q2 total expenses:", np.sum(total_expenses[:3]), "vs", np.sum(total_expenses[3:6]))

annual_inflation = 0.03
monthly_inflation_factor = (1 + annual_inflation) ** (1/12)
inflation_adj_food = food * monthly_inflation_factor ** np.arange(12)
print("\nInflation-adjusted food (first 6):", np.round(inflation_adj_food[:6],2))

n_stocks = 3
stock_base = np.array([100.0, 50.0, 200.0])

monthly_returns_true = np.random.normal(0.01, 0.05, size=(12, n_stocks))
stock_prices = np.empty((12, n_stocks), dtype=np.float64)
stock_prices[0] = stock_base
for t in range(1, 12):
    stock_prices[t] = stock_prices[t-1] * (1 + monthly_returns_true[t])

stock_df = pd.DataFrame(stock_prices, index=months, columns=[f"Stock_{i+1}" for i in range(n_stocks)])
display(stock_df)

stock_returns = (stock_prices[1:] - stock_prices[:-1]) / stock_prices[:-1]
print("\nStock returns shape:", stock_returns.shape)
print("Stock returns (first 3 rows):\n", np.round(stock_returns[:3],4))

print("\nAggregations:")
print("Total yearly savings:", np.round(np.sum(savings),2))
print("Mean monthly spend (food+rent+travel):", np.round(np.mean(total_expenses),2))
print("Std dev monthly spend:", np.round(np.std(total_expenses),2))

weights = np.array([0.4, 0.3, 0.3])
# Use last month's prices as current prices, and compute portfolio value (weights dot prices)
current_prices = stock_prices[-1]  # shape (3,)
portfolio_value = weights.dot(current_prices)
print("\nPortfolio current prices:", np.round(current_prices,2))
print("Portfolio value (per 1 share unit of weights):", np.round(portfolio_value,2))

holdings = np.array([10, 20, 5])  # number of shares
portfolio_value_holdings = holdings.dot(current_prices)
print("Portfolio value (with holdings):", np.round(portfolio_value_holdings,2))

log_returns = np.log(stock_prices[1:] / stock_prices[:-1])
mean_lr = np.mean(log_returns, axis=0)
cov_lr = np.cov(log_returns.T)

n_sims = 1000
n_periods = 1  # one-month ahead
simulated_portfolio_returns = np.empty(n_sims)
rng = np.random.default_rng(12345)
for i in range(n_sims):
    sim = rng.multivariate_normal(mean_lr, cov_lr, size=n_periods)
    # convert log return to simple return approx
    sim_simple = np.exp(sim).prod(axis=0) - 1
    simulated_portfolio_returns[i] = np.dot(weights, sim_simple)

print("\nMonte Carlo simulated portfolio returns (first 6):", np.round(simulated_portfolio_returns[:6],4))
print("MC mean return:", np.round(simulated_portfolio_returns.mean(),4), "std:", np.round(simulated_portfolio_returns.std(),4))

plt.figure()
plt.hist(simulated_portfolio_returns, bins=30)
plt.title("Monte Carlo: Simulated 1-month Portfolio Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.show()

finance_matrix = np.column_stack([income, food, rent, travel])
print("\nfinance_matrix shape:", finance_matrix.shape)

income_col, expenses_cols = finance_matrix[:,0], finance_matrix[:,1:]
print("income_col shape:", income_col.shape, "expenses_cols shape:", expenses_cols.shape)

stock_prices_with_nans = stock_prices.copy()
nan_indices = [(2,1), (7,0), (10,2)]  # (month_index, stock_index)
for idx in nan_indices:
    stock_prices_with_nans[idx] = np.nan

print("\nStock prices snippet with NaNs at positions:", nan_indices)
stock_prices_with_nans_df = pd.DataFrame(stock_prices_with_nans, index=months, columns=[f"Stock_{i+1}" for i in range(n_stocks)])
display(stock_prices_with_nans_df)

mask = np.isnan(stock_prices_with_nans)
masked_prices = np.ma.array(stock_prices_with_nans, mask=mask)
print("Masked array example (month 3):", masked_prices[2])

col_mean_masked = masked_prices.mean(axis=0).filled(np.nan)
print("Masked column means (NaN-aware):", np.round(col_mean_masked,4))

stock1 = stock_prices[:,0]

def moving_average_python(arr, window=3):
    n = len(arr)
    out = []
    for i in range(n - window + 1):
        s = 0.0
        for j in range(window):
            s += arr[i+j]
        out.append(s/window)
    return np.array(out)

def moving_average_numpy(arr, window=3):
    kernel = np.ones(window)/window
    # use 'valid' to mimic the python loop length
    return np.convolve(arr, kernel, mode='valid')

repeats = 2000
start = time.perf_counter()
for _ in range(repeats):
    _ = moving_average_python(stock1, window=3)
py_time = time.perf_counter() - start

start = time.perf_counter()
for _ in range(repeats):
    _ = moving_average_numpy(stock1, window=3)
np_time = time.perf_counter() - start

print("\nPerformance ({} repeats) -> python loop: {:.4f}s, numpy vectorized: {:.4f}s".format(repeats, py_time, np_time))
print("Speedup factor (python / numpy):", round(py_time/np_time, 2))

ma_np = moving_average_numpy(stock1, window=3)
plt.figure()
plt.plot(np.arange(len(stock1)), stock1, label="stock1 prices")
plt.plot(np.arange(3-1, 3-1+len(ma_np)), ma_np, label="3-month MA")
plt.title("Stock1: Price vs 3-month Moving Average")
plt.xlabel("Month index")
plt.ylabel("Price")
plt.legend()
plt.show()

arr64 = np.random.rand(1000000).astype(np.float64)
arr32 = arr64.astype(np.float32)

print("\nDtype and memory: arr64.itemsize:", arr64.itemsize, "arr32.itemsize:", arr32.itemsize)
print("arr64.nbytes:", arr64.nbytes, "arr32.nbytes:", arr32.nbytes)

start = time.perf_counter()
_ = arr64 * 1.000001
t64 = time.perf_counter() - start
start = time.perf_counter()
_ = arr32 * 1.000001
t32 = time.perf_counter() - start
print("Timing of multiply (float64): {:.6f}s, (float32): {:.6f}s".format(t64, t32))

cov_matrix = np.cov(stock_returns.T)
eigvals, eigvecs = np.linalg.eig(cov_matrix)
print("\nCovariance matrix:\n", np.round(cov_matrix,6))
print("Eigenvalues:", np.round(eigvals,6))

summary = {
    "total_yearly_income": [np.round(np.sum(income),2)],
    "total_yearly_expenses": [np.round(np.sum(total_expenses),2)],
    "total_yearly_savings": [np.round(np.sum(savings),2)],
    "avg_monthly_expense": [np.round(np.mean(total_expenses),2)],
    "portfolio_value_holdings": [np.round(portfolio_value_holdings,2)]
}
summary_df = pd.DataFrame(summary)
display(summary_df)

print("\nProject run finished — we've covered Basic → Advanced NumPy concepts interactively.")

