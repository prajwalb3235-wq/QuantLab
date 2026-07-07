import pandas as pd

from quantlab.statistics import (
    calculate_simple_returns,
    calculate_log_returns,
    annualized_returns,
    calculate_covariance,
    annualized_covariance
)

prices = pd.DataFrame(
    {
        "AAPL": [100, 102, 101, 105],
        "MSFT": [50, 52, 53, 55],
    }
)

simple = calculate_simple_returns(prices)
log = calculate_log_returns(prices)

print("Simple Returns")
print(simple)

print("\nLog Returns")
print(log)

print("\nAnnualized Returns")
print(annualized_returns(simple))

print("covariance:", calculate_covariance(log))
print("Annualized Covariance", annualized_covariance(log))