"""
QuantLab

Strategy Backtesting Module

Implements:
- SMA Strategy
- EMA Strategy
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Transaction Cost Analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Running QuantLab Strategy Backtesting Module...")

np.random.seed(42)

num_days = 300  # days
returns = np.random.normal(0.0005, 0.01, num_days)
price = 100 * np.exp(np.cumsum(returns))

df = pd.DataFrame({
    "Price": price
})
df["SMA_20"] = df["Price"].rolling(window=20).mean()
df["SMA_50"] = df["Price"].rolling(window=50).mean()
df["EMA_20"] = df["Price"].ewm(span=20, adjust=False).mean()
df["EMA_50"] = df["Price"].ewm(span=50, adjust=False).mean()

df["signal"]=0
df.loc[df["SMA_20"]>df["SMA_50"],"signal"]=1

df["returns"]=np.log(df["Price"]/df["Price"].shift(1))
df["strategy_returns"]=df["signal"].shift(1) * df["returns"]
df["cum_market"]=np.exp(df["returns"].cumsum())
df["cum_strategy"]=np.exp(df["strategy_returns"].cumsum())

def annualized_ret(returns, periods=252):
    return np.exp(returns.mean() * periods)-1
def annualized_vol(returns,periods=252):
    return returns.std() * np.sqrt(periods)
def annual_sharpe(returns,rf=0.0, periods=252):
    excess=returns-rf/periods
    return excess.mean() / excess.std() * np.sqrt(periods)
def max_drawdown(cum_returns):
    peak=cum_returns.cummax()
    dd=(cum_returns-peak)/peak
    return dd.min()

df["Trade"]=df["signal"].diff().abs()
cost=0.001
df["strategy_ret_cost"]=(df["strategy_returns"]-df["Trade"].shift(1) * cost)
df["cum_strategy_cost"]=np.exp(df["strategy_ret_cost"].cumsum())

market_ret=df["returns"].dropna()
market_cum=df["cum_market"].dropna()
print("Buy and Hold")
print("Return:",annualized_ret(market_ret))
print("volatility",annualized_vol(market_ret))
print("sharpe ratio",annual_sharpe(market_ret))
print("maximum drawdown",max_drawdown(market_cum))

sma_ret=df["strategy_returns"].dropna()
sma_cum=df["cum_strategy"].dropna()
sma_trades=df["signal"].diff().abs().sum()
print("\nSMA strategy")
print("Return:",annualized_ret(sma_ret))
print("volatility",annualized_vol(sma_ret))
print("sharpe ratio",annual_sharpe(sma_ret))
print("max DD",max_drawdown(sma_cum))
print("Trades",sma_trades)

df["EMA_signal"]=(df["EMA_20"]>df["EMA_50"]).astype(int)
df["EMA_returns"]=df["EMA_signal"].shift(1) * df["returns"]
df["EMA_cum"]=np.exp(df["EMA_returns"].cumsum())

ema_ret=df["EMA_returns"].dropna()
ema_cum=df["EMA_cum"].dropna()
ema_trades=df["EMA_signal"].diff().abs().sum()
print("\nEMA strategy")
print("Return:",annualized_ret(ema_ret))
print("volatility",annualized_vol(ema_ret))
print("sharpe ratio",annual_sharpe(ema_ret))
print("max DD",max_drawdown(ema_cum))
print("Trades",ema_trades)

df["ema_trade"] = df["EMA_signal"].diff().abs()
df["ema_ret_cost"] = (
    df["EMA_returns"] - df["ema_trade"].shift(1) * cost
)
df["ema_cum_cost"] = np.exp(df["ema_ret_cost"].cumsum())
print(df["ema_cum_cost"])
