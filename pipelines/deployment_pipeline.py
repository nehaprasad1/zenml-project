import json
import numpy as np
import pandas as pd
import mlflow
from typing import Any
from pydantic import BaseModel

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

# Internal project imports
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from steps.config import ModelNameConfig
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])

# --- CONFIGURATION ---

class DeploymentTriggerConfig(BaseModel):
    """Configuration for the deployment trigger."""
    min_accuracy: float = 0.01

# --- STEPS ---

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Fetches test data for inference as a JSON string."""
    data = get_data_for_test()
    return data

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Decides if the model accuracy meets the threshold."""
    return accuracy >= config.min_accuracy

@step(enable_cache=False)
def custom_mlflow_deployer(
    deploy_decision: bool
) -> str:
    """Custom step to handle deployment on Windows by fetching the latest MLflow run."""
    if deploy_decision:
        # 1. Connect to the MLflow client
        client = mlflow.tracking.MlflowClient()
        
        # 2. Match the experiment name from your logs
        exp_name = "mlflow_example_pipeline"
        experiment = client.get_experiment_by_name(exp_name)
        
        if experiment:
            # 3. Get the latest successful run
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=1,
                order_by=["attributes.start_time DESC"]
            )
            
            if runs:
                run_id = runs[0].info.run_id
                model_uri = f"runs:/{run_id}/model"
                print("\n" + "="*60)
                print(f"🚀 DEPLOYMENT READY (WINDOWS WORKAROUND)")
                print(f"Run ID: {run_id}")
                print(f"Model URI: {model_uri}")
                print("\nTo start your prediction server, run this in a NEW terminal:")
                print(f"mlflow models serve -m \"{model_uri}\" --port 8000")
                print("="*60 + "\n")
                return model_uri
        
        print(f"❌ Could not find experiment '{exp_name}' or a recent run.")
        return "None"
    else:
        print("⚠️ Model accuracy too low. Deployment skipped.")
        return "None"

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Finds the MLflow deployment service."""
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=False, 
    )

    if not services:
        raise RuntimeError(f"No MLflow prediction service found for {pipeline_name}.")
    return services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str, 
) -> np.ndarray:
    """Runs a prediction request."""
    service.start(timeout=15) 
    data_dict = json.loads(data)
    data_dict.pop("columns", None)
    data_dict.pop("index", None)
    
    columns_for_df = [
        "payment_sequential", "payment_installments", "payment_value",
        "price", "freight_value", "product_name_lenght",
        "product_description_lenght", "product_photos_qty", "product_weight_g",
        "product_length_cm", "product_height_cm", "product_width_cm",
    ]
    
    df = pd.DataFrame(data_dict["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    final_data = np.array(json_list)
    
    prediction = service.predict(final_data)
    return prediction

# --- PIPELINES ---

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str, 
    min_accuracy: float = 0.01,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    """Pipeline that trains and triggers custom deployment."""
    # 1. Ingest
    df = ingest_df(data_path)
    
    # 2. Clean
    X_train, X_test, y_train, y_test = clean_df(df)
    
    # 3. Train
    config = ModelNameConfig()
    model = train_model(X_train, X_test, y_train, y_test, config=config)
    
    # 4. Evaluate
    r2_score, mse = evaluate_model(model, X_test, y_test)
    
    # 5. Trigger logic
    trigger_config = DeploymentTriggerConfig(min_accuracy=min_accuracy)
    decision = deployment_trigger(accuracy=r2_score, config=trigger_config)
    
    # 6. Custom Deployer (Windows Friendly)
    custom_mlflow_deployer(deploy_decision=decision)

@pipeline(enable_cache=False)
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    """Pipeline for batch inference logic."""
    batch_data = dynamic_importer()
    print("Inference data prepared. Please ensure the MLflow server is running manually.")