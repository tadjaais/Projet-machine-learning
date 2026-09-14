"""Pipeline de préparation des données de ventes Fibre."""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def clean_debit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['debit'] = df['debit'].astype(str).str.strip()
    df['debit'] = df['debit'].replace(r'(?i)\b1gb\b', '1GB', regex=True)
    df['debit'] = df['debit'].replace(['', 'nan', 'None'], '50M')
    df['debit'] = df['debit'].fillna('50M')
    return df


def clean_profil(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['profil'] = df['profil'].fillna('Non renseigné')
    return df


def add_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['mois'] = df['mois'].astype(str).str[-2:]
    df['date'] = pd.to_datetime(df['Année'].astype(str) + df['mois'], format='%Y%m')
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_debit(df)
    df = clean_profil(df)
    df = add_date_column(df)
    return df


def split_train_test(df: pd.DataFrame, cutoff: str = '2025-01-01'):
    train_df = df[df['date'] < cutoff]
    test_df = df[df['date'] >= cutoff]
    return train_df, test_df


def aggregate_monthly(df: pd.DataFrame, target_col: str = 'Nbre recrutements') -> pd.Series:
    return df.groupby(pd.Grouper(key='date', freq='M'))[target_col].sum()
