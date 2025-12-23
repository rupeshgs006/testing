from fastapi import FastAPI
from app.schema import BatchPredictInput, BatchPredictOutput 
from  app.model import ml_model
from pydantic import BaseModel
from fastapi import HTTPException
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# @app.post("/square")
# def square(x: float):
#     return {"result ": x*x}


# class SquareInput(BaseModel):
#     x:float

# @app.post("/square")
# def square(data:SquareInput):
#     return {"result ": data.x * data.x}

# @app.post("/predcit",response_model=PredictOutput)
# def predict(data:PredictInput):
#     result = power_pred(data.x,data.power)
#     return {"result": result}


# @app.post("/predict", response_model=PredictOutput)
# def predict(data: PredictInput):
#     try:
#         y = ml_model.predict(data.x)
#         return {"y": y}
#     except Exception as e:
#         logger.exception("Prediction failed")
#         raise HTTPException(status_code=500, detail="Prediction error")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_batch",response_model=BatchPredictOutput)
def predict_batch(data:BatchPredictInput):
    ys = ml_model.predict_batch(data.xs)
    return {"ys":ys}
