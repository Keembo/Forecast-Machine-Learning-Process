# Import modules
from fastapi import FastAPI, Request
from joblib import load
import pandas as pd
import uvicorn

# App definition
app = FastAPI()

# Api Status
@app.get("/")
async def hello():
    response = {
        "status": 200,
        "message": "It Works!"
    }
    return response

# Load model function
def load_model():
    with open(r'model/demand_forecasting_model.pkl', 'rb') as file:
        model = load(file)
    return model

# Predictions Function
def make_predictions(model, year, month):
    input_data_time = pd.to_datetime(f"{year}-{month}-01") 
    months_ahead = (input_data_time.year - 2017) * 12 + input_data_time.month - 12 
    
    start = len(model.model.endog)
    end = start + months_ahead - 1
    
    predictions = model.predict(start=start, end=end, dynamic=False)
    predictions = predictions.iloc[-1]
    return predictions

# Check model status
@app.get("/model-status")
async def check_model():
    model = load_model()
    if model is not None:
        response = {
            "status": 200,
            "message": "The Model also works!"
        }
    else:
        response = {
            "status": 404,
            "message": "Model Not Found!"
        }
    return response

# Predictons with api
@app.post("/predict")
async def predict(request: Request):
    # get data from request
    data = await request.json()
    
    # data
    month_request = data['order_month']
    year_request = data['order_year']
    
    # load model
    model = load_model()
    
    # validation to make sure the request is above training data
    if month_request < 12 and year_request < 2017:
        response = {
            "code": 400,
            "message": "Month and Year needs to be greater than 12 and greater than 2017!"
        }
        return response
    
    # prediction
    try :
        prediction = make_predictions(model, year_request, month_request)
        
        response = {
            "code": 200,
            "message": "Succes",
            "prediction": "Forecast for month: " + str(month_request) + ", year: " + str(year_request) + " is: " + str(prediction)
        }
    except:
        response = {
            "code": 404,
            "message": "Prediction Failed!"
        }
    return response

# Main
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)