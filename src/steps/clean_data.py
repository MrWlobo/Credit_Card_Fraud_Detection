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
        try:
            float_cols = [c for c in df.columns if c != "Class"]

            logging.info("Correcting data types")
            if "Class" in df.columns:
                df["Class"] = df["Class"].astype("int32")
                
            if float_cols:
                df[float_cols] = df[float_cols].astype("float64")
                
            return df
        except Exception as e:
            logging.error(f"Error while correcting data types: {e}")
            raise e

class DataHandleNullValues(DataCleaningStrategy):
    @staticmethod
    def handle_data(df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Handling null values")
            return df.fillna(df.mean(numeric_only=True))
        except Exception as e:
            logging.error(f"Error while handling null values: {e}")
            raise e


@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Cleaning data")
        data_cleaning_sub_steps = [
            DataHandleTypes,
            DataHandleNullValues,
        ]
        for sub_step in data_cleaning_sub_steps:
            df = sub_step.handle_data(df)

        return df
    except Exception as e:
        logging.error(f"Error while cleaning data: {e}")
        raise e
