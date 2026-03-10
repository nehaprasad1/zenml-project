from pipelines.train_pipeline import train_pipeline
from zenml.client import Client
if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path=r"E:\mlops\znml\data\data.csv")
#mlflow ui --backend-store-uri 