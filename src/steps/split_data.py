import logging
from typing import Annotated, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from zenml import step
import torch
from torch.utils.data import Dataset


class CreditCardFraudDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

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

        feature_columns = [col for col in df.columns if col != "Class"]

        logging.info("Scaling the features")
        scaler = StandardScaler()
        train_features = scaler.fit_transform(df_train[feature_columns])
        val_features = scaler.transform(df_val[feature_columns])
        test_features = scaler.transform(df_test[feature_columns])

        logging.info("Getting the labels")
        train_targets = df_train["Class"].values
        val_targets = df_val["Class"].values
        test_targets = df_test["Class"].values

        logging.info("Assembling datasets")
        train_dataset = CreditCardFraudDataset(train_features, train_targets)
        val_dataset = CreditCardFraudDataset(val_features, val_targets)
        test_dataset = CreditCardFraudDataset(test_features, test_targets)

        return train_dataset, val_dataset, test_dataset

    except Exception as e:
        logging.error(f"Error while splitting the data: {e}")
        raise e
    