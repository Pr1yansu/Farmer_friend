from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np

model_filename = 'crop_recommendation.joblib'
scaler_model = 'Scaling_Model.joblib'
loaded_model = load(model_filename)
loaded_scaler = load(scaler_model)

app = FastAPI()


class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.post("/predict_crop")
async def predict_crop(input_data: CropInput):
    arr = np.array([[input_data.N, input_data.P, input_data.K, input_data.temperature,
                   input_data.humidity, input_data.ph, input_data.rainfall]])
    arr = loaded_scaler.transform(arr)
    prediction = loaded_model.predict(arr)
    return {"predicted_crop": prediction[0]}
