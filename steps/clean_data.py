import logging
from zenml import step
import pandas as pd

from src.data_cleaning import DataPreProcessStrategy , DataCleaner , DataDivideStrategy 
from typing_extensions import Annotated
from typing import Tuple
@step
def clean_df(df:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame,"y_train"],
    Annotated[pd.DataFrame,"y_test"]
]:
    """
    CLean the data and split it into training and test data
    Args:
        df: raw data
    Returns:
        X_train :training data,
        X_test : testing data ,
        y_test : tessting labels,
        y_train: traing lables

    """
    try:
        process_Strategy = DataPreProcessStrategy
        data_cleaning = DataCleaner(df,process_Strategy)
        processed_data= data_cleaning.handle_data()
        
        divide_strategy =  DataDivideStrategy()
        data_cleaning = DataCleaner(processed_data,divide_strategy)
        X_train , X_test , y_train ,y_test =  data_cleaning.handle_data()
        logging.info("Data Cleaning has completed")
    except Exception as e:
        logging.error("Error while cleaining data: {}".formet(e))
