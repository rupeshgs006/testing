from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import List

class BatchPredictInput(BaseModel):
    # x:float
    xs : List[float] = Field(...,min_length=1)

class BatchPredictOutput(BaseModel):
    # y:float
    ys:List[float]

