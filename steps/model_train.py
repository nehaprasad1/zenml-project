import logging

import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.model_development import LinearRegressionModel
from .config import ModelNameConfig

@step
def train_model(
    X_train :pd.DataFrame,
    X_test:pd.DataFrame,
    y_train:pd.DataFrame,
    y_test:pd.DataFrame,
    config:ModelNameConfig
) -> RegressorMixin :
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        
        model = None
        if config.model_name== "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train,y_train)
            return trained_model
        else:
            raise ValueError("Model {} not listed".format(config.model_name))
    except Exception as e:
        logging.error("error while model trainig step : {}".format(e))
        raise e