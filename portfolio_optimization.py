"""
QuantLab

Portfolio Optimization Module

Implements:
- Log Return Computation
- Covariance Matrix
- Minimum Variance Portfolio
- Efficient Frontier
- Maximum Sharpe Portfolio
- PCA Analysis
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Running QuantLab Portfolio Optimization Module...")
    
tickers=[
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS"
]
start_date="2022-01-01"
end_date="2024-01-01"
prices=yf.download(
    tickers,
    start=start_date,
    end=end_date
)["Close"]
print(prices.head())
log_returns=np.log(prices/prices.shift(1)).dropna()
print('log returns:',log_returns.head())
#expected_returns vector. basically average returns of that stock over the given time period
expected_returns=log_returns.mean().values
print('expected returns',expected_returns)
#covariance_matrix refers to the covariance matrix whose diagonal elements refers to the variance of each stock and other elements refer to the relation between each stock returns.symmetric matrix
covariance_matrix=np.cov(log_returns,rowvar=False)
print('covariance_matrix',covariance_matrix)
n=len(expected_returns)
#w-matrix of size n*n with all entries as 1
w=np.ones(n)/n
portfolio_returns=w @ expected_returns
portfolio_variance=w.T @ covariance_matrix @ w
portfolio_risk=np.sqrt(portfolio_variance)
sharpe_ratio= portfolio_returns/portfolio_risk
print('expected sharpe',sharpe_ratio)
log_returns.plot(figsize=(10,5))
plt.title("Log Returns")
plt.show()
#eigvecs-eigen vectors matrix which capture variance from highest to lowest. eigvals-corresponding measure of variance measured by that particular vector
eigvals,eigvecs=np.linalg.eigh(covariance_matrix)
idx=np.argsort(eigvals)[::-1]
eigvals=eigvals[idx]
eigvecs=eigvecs[:,idx]
explained_variance_ratio=eigvals/eigvals.sum()
print('explained variance ratio:',explained_variance_ratio)
pca_returns=log_returns @ eigvecs
print("pca returns:",pca_returns)
print(eigvecs[:,1])
n1=covariance_matrix.shape[0]
unit_vector=np.ones(n1)
covariance_matrix_inv=np.linalg.inv(covariance_matrix)
#minimum_variance_weights is the weight for MVP. 
minimum_variance_weights=covariance_matrix_inv @ unit_vector
minimum_variance_weights=minimum_variance_weights/(unit_vector.T @ covariance_matrix_inv @ unit_vector)
print('MVP weights vector:',minimum_variance_weights)
port_var=minimum_variance_weights.T @ covariance_matrix @ minimum_variance_weights
print(portfolio_variance)
print(port_var)
#portfolio_returns=w.T @ expected_returns where w is the portfolio weights and expected_returns is the expected returns vector
#portfolio_variance= w.t @ covariance_matrix @ w where covariance_matrix is the covariance matrix of log_returns and minimizing this gives us port_var that is MVP(minimum variance portfolio) meaning least possible risk
# we use the constraint unit_vector.T @ w=1 cuz this means invest fully. when we talk abt maximum return focussed only, we mean invest fully in the one that gives max returns in the expected return vector which isnt sensible cuz it doesnt take into account the risk.
# hence, we gotta make a choice that takes both, risk and returns into account.
#r_vals refers to the target return values for which we'll calculate the lowest possible risk, so tht we can make an educated choice. covariance_matrix_vals r the corresponding risk of these r_vals.
r_vals=np.linspace(expected_returns.min(),expected_returns.max(),50)
# linspace just picks numbers from expected_returns.min() to expected_returns.max() with 50 numbers overall equally spaced including first and last num. for some r, we need w.T @ expected_returns=r and unit_vector.T @ w=1 
coeff_A=unit_vector.T @ covariance_matrix_inv @ unit_vector
coeff_B=unit_vector.T @ covariance_matrix_inv @ expected_returns
coeff_C=expected_returns.T @ covariance_matrix_inv @ expected_returns
r1=r_vals[10]
var_r=coeff_A*(r1**2) - 2*coeff_B*r1 + coeff_C
var_r=var_r/(coeff_A*coeff_C - coeff_B**2)
covariance_matrix_v=np.sqrt(var_r)
covariance_matrix_vals=[]
for rx in r_vals:
    vary_r=(coeff_A*(rx**2) - 2*coeff_B*rx + coeff_C)/(coeff_A*coeff_C - coeff_B**2)
    covariance_matrix_vals.append(np.sqrt(vary_r))
covariance_matrix_vals=np.array(covariance_matrix_vals)    
r_mvp = minimum_variance_weights.T @ expected_returns 
mask = r_vals >= r_mvp
r_eff = r_vals[mask]
covariance_matrix_eff = covariance_matrix_vals[mask]
plt.plot(covariance_matrix_eff, r_eff)
plt.scatter(np.sqrt(1/coeff_A), r_mvp, color='red')
plt.xlabel("Risk")
plt.ylabel("Expected Return")
plt.grid(True)
plt.show()
#maximizing sharpe ratio is our next step. the formula is given by (w.T @ expected_returns)/np.sqrt(w.T @ covariance_matrix @ w). note that sharpe ratio is scale invariant, so we just maximize the numerator and keep denominator as 1 for simplicity and as a constraint.
#applying langrangian and all that, we get weights vector to be proportional to covariance_matrix_inv @ expected_returns.after normalizing we get w_sh=(covariance_matrix_inv @ expected_returns)/(unit_vector.T @ covariance_matrix @ expected_returns)
print("expected_returns.min():", expected_returns.min())
print("expected_returns.max():", expected_returns.max())
print("r_mvp:", coeff_B / coeff_A)
print("MVP expected return:", r_mvp)
maximum_sharpe_weights=covariance_matrix_inv @ expected_returns
maximum_sharpe_weights=maximum_sharpe_weights/(unit_vector.T @ covariance_matrix_inv @ expected_returns)
expected_returns_ms=maximum_sharpe_weights @ expected_returns
risk_ms=np.sqrt(maximum_sharpe_weights.T @ covariance_matrix @ maximum_sharpe_weights)
sharpe_ms=expected_returns_ms/risk_ms
print('return', expected_returns_ms)
print('risk', risk_ms)
print('new sharpe ratio', sharpe_ms)
sharpe_vals = r_eff / covariance_matrix_eff
idx = np.argmax(sharpe_vals)

r_ms = r_eff[idx]
covariance_matrix_ms = covariance_matrix_eff[idx]

plt.plot(covariance_matrix_eff, r_eff, label="Efficient Frontier")
plt.scatter(np.sqrt(port_var), r_mvp, color="red", label="MVP")
plt.scatter(covariance_matrix_ms, r_ms, color="green", label="Max Sharpe")
plt.legend()
plt.grid(True)
plt.show()

