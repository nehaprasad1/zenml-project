#from zenml.steps import BaseParameters
"""model configs
"""
#class ModelNameConfig(BaseParameters):
#    model_name: str="LinearRegression"
from pydantic import BaseModel
from zenml import step

# Define  config using Pydantic
class ModelNameConfig(BaseModel):
    model_name: str = "LinearRegression"    