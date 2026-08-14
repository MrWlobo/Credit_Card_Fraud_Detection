import logging
from typing import Annotated, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step


@step
def split_data(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_val"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_val"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        X = df.drop(columns=["Class"])
        y = df["Class"]

        logging.info("Splitting the dataset into training and testing sets")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        logging.info("Getting the validation set from training set")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.1765,
            random_state=42,
            stratify=y_train_val,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    except Exception as e:
        logging.error(f"Error while splitting the data: {e}")
        raise e
    