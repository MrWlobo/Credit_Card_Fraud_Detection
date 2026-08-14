import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step
from typing import Tuple


@step
def split_data(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
