import logging
import pandas as pd
from zenml import step

@step
def train_model(X_train: pd.DataFrame, y_train) -> None:
    pass