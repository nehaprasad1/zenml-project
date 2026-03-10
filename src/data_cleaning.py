import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1)
            cols_to_median = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
            for col in cols_to_median:
                if col in data.columns:
                    data[col] = data[col].fillna(data[col].median())

            # 2. Handle the text column
            if "review_comment_message" in data.columns:
                data["review_comment_message"] = data["review_comment_message"].fillna("No review")
            
            
            data = data.select_dtypes(include=[np.number])
            data = data.fillna(data.median())
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
            return data
        
        except Exception as e:
            logging.error("Error in preprocesing data: {}".format(e))
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            if "review_score" not in data.columns:
                raise ValueError("Target column 'review_score' not found in numeric columns.")
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data:{}".format(e))
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) :
        """Initializes the DataCleaning class with a specific strategy."""
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e
#if __name__ == "__main__":
#    data = pd.read_csv("E:\mlops\znml\data\olist_customers_dataset.csv")
#    data_cleaning = DataCleaning(data,DataPreprocessStrategy())
#    data_cleaning.handle_data()
