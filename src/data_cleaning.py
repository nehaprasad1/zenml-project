import logging
from abc import ABC , abstractmethod
from typing import Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    abstract class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame , pd.Series]:
        pass
class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for data pre-processing
    """
    def handle_data(self, data)-> pd.DataFrame:
        """
        preprocess data

        Args:
            data (_type_): _description_

        Returns:
            pd.DataFrame: _description_
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
            #fill up the null value with median 
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True)
            #drop numeric data
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error("Error in Pre-Processing data: {}".format(e))
            raise e
class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test

    """
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame , pd.Series]:
        """ 
        divide data into train and test

        Args:
            data (pd.DataFrame): _description_

        Returns:
            Union[pd.DataFrame , pd.Series]: _description_
        """
        try:
            X = data.drop(["review_score"],axis=1)
            y = data["review_score"]
            X_train , X_test ,y_train ,y_test = train_test_split(X,y,test_size =0.2,random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing Data : {} ".formet(e))
            raise e
    class DataCleaner:
        """
        Cleaining data and splittig
        """
        def __init__(self,  data: pd.DataFrame,strategy:DataStrategy):
            self.data = data
            self.strategy = strategy
        def handle_data(self)->Union[pd.DataFrame,pd.Series]:
            """Handle Data"""
            try:
                return self.strategy.handle_data(self.data)
            except Exception as e:
                logging.error("Error in handling data{}".format(e))
                raise e
                        