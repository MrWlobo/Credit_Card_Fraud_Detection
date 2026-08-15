import logging
from typing import Annotated, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step
import torch
from torch.utils.data import Dataset


class CreditCardFraudDataset(Dataset):
    def __init__(self, df):
        self.features = torch.tensor(df.drop(columns=["Class"]).values, dtype=torch.float32)
        self.targets = torch.tensor(df["Class"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


@step
def split_data(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[Dataset, "train_dataset"],
    Annotated[Dataset, "val_dataset"],
    Annotated[Dataset, "test_dataset"],
]:
    try:
        logging.info("Splitting the dataset into training, validation and testing sets")
        df_train, df_temp = train_test_split(df, test_size=0.4, stratify=df["Class"], random_state=42)
        df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp["Class"], random_state=42)

        train_dataset = CreditCardFraudDataset(df_train)
        val_dataset = CreditCardFraudDataset(df_val)
        test_dataset = CreditCardFraudDataset(df_test)

        return train_dataset, val_dataset, test_dataset

    except Exception as e:
        logging.error(f"Error while splitting the data: {e}")
        raise e
    