from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.utils.data_utils import encoding_test, feature_engineering, imputation, categorize_flight_time, handlapitest

MODEL_PICKEL_PATH = Path('models/model.joblib').absolute()
DATA_SCHEMA_PATH = Path("data/data_transformation/test.csv").absolute()

# Define FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load(MODEL_PICKEL_PATH)

# Define Pydantic model for input data
class InputData(BaseModel):
    num_passengers: int
    channel: str
    trip: str
    purchase_lead: int
    length_of_stay: int
    flight_hour: int
    day: str
    route: str
    booking_origin: str
    wants_extra_baggage: int
    wants_preferred_seat: int
    wants_in_flight_meals: int
    flight_duration: float



# Endpoint to predict
@app.post("/predict")
async def pred(data:InputData):
    # Convert input data to pandas DataFrame
    df = pd.DataFrame(data.dict(), index=[0])
   
    # Preprocessing steps
    
        
    df_apitest= handlapitest(df, DATA_SCHEMA_PATH)
    df_apitest = encoding_test(df_apitest, categorize_flight_time)
    df_apitest = feature_engineering(df_apitest)

    pred = model.predict(np.array(df_apitest))  
    resp = {'pred': pred.tolist()[0]}

    return resp
