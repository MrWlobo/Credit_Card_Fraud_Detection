from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.split_data import split_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

@pipeline
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    cleaned_df = clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(cleaned_df)
    train_model()
    evaluate_model()
