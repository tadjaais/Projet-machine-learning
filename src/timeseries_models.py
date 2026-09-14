"""Modèles de séries temporelles : ARIMA, SARIMA, Holt-Winters."""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


def test_stationarity(ts: pd.Series) -> dict:
    stat, p_value, _, _, critical_values, _ = adfuller(ts)
    return {'statistic': stat, 'p_value': p_value, 'critical_values': critical_values}


def decompose(ts: pd.Series, model: str = 'additive'):
    return seasonal_decompose(ts, model=model)


def fit_arima(ts: pd.Series, order: tuple):
    return ARIMA(ts, order=order).fit()


def fit_sarima(ts: pd.Series, order: tuple, seasonal_order: tuple):
    model = SARIMAX(
        ts,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def fit_holt_winters(ts: pd.Series, trend: str = 'add', seasonal: str = 'mul', seasonal_periods: int = 12):
    model = ExponentialSmoothing(ts, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    return model.fit()
