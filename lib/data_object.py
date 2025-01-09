from pydantic import BaseModel

# Request body model for loading a model
class LoadModelRequest(BaseModel):
    models_name: str
    
# Request for loading new translate method
class LoadMethodRequest(BaseModel):
    method_name: str

