from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np

# ! For crop Recommendation
model_filename = 'crop_recommendation.joblib'
scaler_model = 'Scaling_Model.joblib'
loaded_model = load(model_filename)
loaded_scaler = load(scaler_model)

# ! For Fertilizer Recommendation
fmodel_filename = 'Fartilizer_model.joblib'
fscaler_filename = 'Fartilizer_Scaler.joblib'
fmodel = load(fmodel_filename)
fscaler = load(fscaler_filename)

app = FastAPI()


class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class FartInput(BaseModel):
    N: float
    P: float
    K: float

@app.post("/predict_crop")
async def predict_crop(input_data: CropInput):
    arr = np.array([[input_data.N, input_data.P, input_data.K, input_data.temperature,input_data.humidity, input_data.ph, input_data.rainfall]])
    arr = loaded_scaler.transform(arr)
    prediction = loaded_model.predict(arr)
    return {"predicted_crop": prediction[0]}

@app.post("/predict_fart")
async def predict_fart(input_data: FartInput):
    arr = np.array([[input_data.N, input_data.K, input_data.P]])
    arr = fscaler.transform(arr)
    prediction = fmodel.predict(arr)
    return {"predicted_fartilizer": prediction[0]}