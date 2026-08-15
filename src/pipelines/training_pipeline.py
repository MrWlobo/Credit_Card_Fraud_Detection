from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.split_data import split_data
from steps.model_handling import train_model, evaluate_model

@pipeline
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    cleaned_df = clean_data(df)
    train_dataset, val_dataset, test_dataset, scaler, feature_count = split_data(cleaned_df)
    model = train_model(train_dataset, val_dataset, feature_count)
    evaluate_model(test_dataset, model)
