import pandas as pd


def calculate_covariance(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the covariance matrix of asset returns.
    """
    return returns.cov()


def annualized_covariance(
    returns: pd.DataFrame,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Compute the annualized covariance matrix.
    """
    return returns.cov() * trading_days