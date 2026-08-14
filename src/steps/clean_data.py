import logging
import pandas as pd
from zenml import step
from abc import ABC, abstractmethod


class DataCleaningStrategy(ABC):
    @abstractmethod
    def handle_data(df: pd.DataFrame) -> pd.DataFrame:
        pass

class DataHandleTypes(DataCleaningStrategy):
    def handle_data(df):
        for column in df.columns:
            if column == "Class":
                df[column] = df[column].astype("int32")
                continue

            df[column] = df[column].astype("float64")
        return df


@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = DataHandleTypes.handle_data(df)
    return df