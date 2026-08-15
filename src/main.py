from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    training_pipeline.with_options(enable_cache=False)(data_path="data/creditcard.csv")
