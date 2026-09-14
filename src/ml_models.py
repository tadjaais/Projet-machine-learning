"""Modèles Machine Learning : Régression Linéaire, Random Forest, XGBoost."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DEFAULT_CATEGORICAL_VARS = [
    'geolocalisaton', 'Localisation', 'canal', 'partenaire',
    'franchise', 'categorie', 'profil', 'debit', 'plan tarifaire',
]


def add_features(df: pd.DataFrame, target_col: str = 'Nbre recrutements',
                  lags=(1, 2, 3), windows=(3, 6)) -> pd.DataFrame:
    df = df.copy()
    df['tendance'] = np.arange(len(df))
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    for window in windows:
        df[f'rolling_{window}'] = df[target_col].rolling(window).mean()
    return df


def encode_categorical(df: pd.DataFrame, categorical_vars=DEFAULT_CATEGORICAL_VARS) -> pd.DataFrame:
    return pd.get_dummies(df, columns=categorical_vars, drop_first=True)


def build_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    df_features = add_features(df_clean)
    df_features = df_features.dropna()
    return encode_categorical(df_features)


def get_default_models() -> dict:
    return {
        'Régression Linéaire': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42),
    }


def train_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results.append({
            'Modèle': name,
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R²': r2_score(y_test, y_pred),
        })
    return pd.DataFrame(results)
