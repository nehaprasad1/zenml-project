from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_evaluation import evaluate_model
from steps.model_train import train_model 
@pipeline(enable_cache=True)
def train_pipeline(data_path:str):
    df = ingest_df(data_path)
    clean_df(df)
    train_model(df)
    