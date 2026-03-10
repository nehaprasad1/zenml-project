import logging
from abc import ABC , abstractmethod
from sklearn.linear_model import LinearRegression
class Model(ABC):
    """
    Absstract clas for all modelss
    """
    @abstractmethod
    def train(self, X_train ,y_train):
        """ trains the ml model

        Args:
            X_train (_DataFrame_) : training data
            y_train (_type_:series): Training labels
        """
        pass
class LinearRegressionModel(Model):
    """Linear Regression Model"""
    def train(self, X_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train,y_train)
            logging.info("Model training completed")
            return  reg
        except Exception as e:
            logging.error("Error while model traing: {}".format(e))
            raise e
    