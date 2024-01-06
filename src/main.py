from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from typing import List
from lib.utils.data import process_data
from lib.utils.model import inference, load_model
import uvicorn
import os

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
model_path = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(), './model/model.pkl')))


# Define a Pydantic model for the input data
class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        allow_population_by_field_name = True

# Create a FastAPI app
app = FastAPI()

# Root endpoint with a welcome message
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Model Inference API!"}

# POST endpoint for model inference
@app.post("/predict")
def predict(data: InputData):
    original_data = data.dict(by_alias=True)
    X, *_ = process_data(original_data, categorical_features=cat_features, label="salary", inference = True)
    logger.info("PROCESS DATA DONEEEEEEEEE")
    model = load_model(model_path)
    prediction = inference(model, X)
    logger.info("PREDICTION DONEEEEEEEEE")

    return {"predictions": [int(prediction)]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
