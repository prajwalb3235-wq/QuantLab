"""
Functions for computing asset returns.

This module provides utilities to calculate simple returns,
logarithmic returns and annualized returns from historical
price data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple percentage returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical asset prices indexed by date.

    Returns
    -------
    pd.DataFrame
        Simple returns for each asset.
    """
    return prices.pct_change().dropna()


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute logarithmic returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical asset prices indexed by date.

    Returns
    -------
    pd.DataFrame
        Logarithmic returns for each asset.
    """
    return np.log(prices / prices.shift(1)).dropna()


def annualized_returns(
    returns: pd.DataFrame,
    trading_days: int = 252,
) -> pd.Series:
    """
    Compute annualized mean returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns.

    trading_days : int, default=252
        Number of trading days in a year.

    Returns
    -------
    pd.Series
        Annualized return of each asset.
    """
    return returns.mean() * trading_days