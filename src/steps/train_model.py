import logging
import pandas as pd
from zenml import step
from abc import ABC, abstractmethod


class CreditCardFraudModel(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
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
    pass