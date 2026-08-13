import logging
import pandas as pd
from zenml import step

@step
def evaluate_model(X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    pass