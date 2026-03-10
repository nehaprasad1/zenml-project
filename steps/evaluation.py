import  logging
from zenml import step
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from src.evaluation import MSE ,R2Score , RMSE
from sklearn.base import RegressorMixin

@step
def evaluate_model(
    model: RegressorMixin, 
    x_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "mse"]]:

    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """
    try:
        
        prediction = model.predict(x_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        r2_class = R2Score()
        r2_score = r2_class.calculate_score(y_test, prediction)
        return r2_score, mse
    except Exception as e:
        logging.error("Error in Evaluating Model step: {}".format(e))
        raise e
