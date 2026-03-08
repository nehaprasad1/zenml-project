import logging
import pandas as pd
from zenml import step

@step
def train_model(
    df:pd.DataFrame
) -> None :
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    pass