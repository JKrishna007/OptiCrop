"""
OptiCrop — Data Preprocessing Module

Handles dataset loading, cleaning, feature scaling, label encoding,
and train/test splitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# ── Feature columns & target ────────────────────────────────────────────────
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL = "label"


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the crop recommendation CSV into a DataFrame.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to `crop_recommendation.csv`.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    required = set(FEATURE_COLS + [TARGET_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, duplicates, and basic type coercion.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame  — cleaned copy
    """
    df = df.copy()

    # Drop exact duplicates
    df.drop_duplicates(inplace=True)

    # Drop rows with any null in the feature / target columns
    df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)

    # Ensure numeric types for features
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=FEATURE_COLS, inplace=True)

    # Strip & lowercase crop labels
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.lower()

    df.reset_index(drop=True, inplace=True)
    return df


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """
    Fit a StandardScaler on `X_train` and transform both splits.

    Returns
    -------
    (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def encode_labels(y: pd.Series):
    """
    Encode string crop labels into integers.

    Returns
    -------
    (y_encoded, label_encoder)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split the DataFrame into features / target and then train / test sets.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def preprocess_pipeline(filepath: str, test_size: float = 0.2):
    """
    End-to-end preprocessing: load → clean → split → encode → scale.

    Returns
    -------
    dict with keys:
        X_train, X_test          — scaled numpy arrays
        y_train, y_test          — encoded numpy arrays
        scaler                   — fitted StandardScaler
        label_encoder            — fitted LabelEncoder
        df                       — cleaned DataFrame
    """
    df = load_data(filepath)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size)

    y_train_enc, le = encode_labels(y_train)
    y_test_enc = le.transform(y_test)

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    return {
        "X_train": X_train_sc,
        "X_test": X_test_sc,
        "y_train": y_train_enc,
        "y_test": y_test_enc,
        "scaler": scaler,
        "label_encoder": le,
        "df": df,
    }
