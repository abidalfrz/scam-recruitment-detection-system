from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import pandas as pd
from services.predictor import predictor
from services.preprocessing import preprocess
from schemas.schema import PredictionModel, PredictionResponse

app = FastAPI(title="Job Scam Detection API")

@app.get("/")
def index():
    return {"message": "Welcome to the Job Scam Detection API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionModel):
    data_dict = data.dict()
    df = pd.DataFrame([data_dict])
    X = preprocess(df)
    prediction = predictor.predict(X)[0]
    prediction_proba = predictor.predict_proba(X)[0][prediction]
    mapping = {0: "Real", 1: "Fraudulent"}
    prediction = mapping[prediction]
    response = {
        "prediction": prediction,
        "probability": float(prediction_proba)
    }
    return response

if __name__ == "__main__":
    FILE_NAME = "app"
    ENTRY_POINT = "app"
    HOST = "127.0.0.1"
    PORT = 8000
    uvicorn.run(f"{FILE_NAME}:{ENTRY_POINT}", host=HOST, port=PORT, reload=True)

