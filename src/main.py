from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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

app = FastAPI()

@app.get("/")
def read_root():
    logger.info("Welcome to the FastAPI Model Inference API!")
    return {"message": "Welcome to the FastAPI Model Inference API!"}

@app.post("/predict")
def predict(data: InputData):
    logger.info("Solving '-' problem...")
    original_data = data.dict(by_alias=True)
    logger.info("Starting preprocessing process...")
    X, *_ = process_data(original_data, categorical_features=cat_features, label="salary", inference = True)
    logger.info("Data processed")
    model = load_model(model_path)
    logger.info("Getting prediction...")
    prediction = inference(model, X)
    logger.info("Prediction is:")
    logger.info(prediction)

    return {"predictions": [int(prediction)]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
