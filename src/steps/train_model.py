import logging
import pandas as pd
from zenml import step
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset, DataLoader


class CreditCardFraudModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def perform_model_training():
        pass

class BasicCreditCardFraudModel(CreditCardFraudModel):
    def __init__(self):
        pass

    def perform_model_training():
        pass


@step
def train_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
     y_val: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
        logging.info("Creating data loaders")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)