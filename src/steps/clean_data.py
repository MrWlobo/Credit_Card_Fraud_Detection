import logging
import pandas as pd
from zenml import step
from abc import ABC, abstractmethod


class DataCleaningStrategy(ABC):
    @staticmethod
    @abstractmethod
    def handle_data(df: pd.DataFrame) -> pd.DataFrame:
        pass

class DataHandleTypes(DataCleaningStrategy):
    @staticmethod
    def handle_data(df: pd.DataFrame) -> pd.DataFrame:
        float_cols = [c for c in df.columns if c != "Class"]
        
        if "Class" in df.columns:
            df["Class"] = df["Class"].astype("int32")
            
        if float_cols:
            df[float_cols] = df[float_cols].astype("float64")
            
        return df

class DataHandleNullValues(DataCleaningStrategy):
    @staticmethod
    def handle_data(df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(df.mean(numeric_only=True))


@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data_cleaning_sub_steps = [
        DataHandleTypes,
        DataHandleNullValues,
    ]
    for sub_step in data_cleaning_sub_steps:
        df = sub_step.handle_data(df)
    return df
