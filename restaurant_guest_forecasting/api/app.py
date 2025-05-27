from fastapi import FastAPI

from restaurant_guest_forecasting.api.input import ModelInput
from restaurant_guest_forecasting.api.models.load_models import load_random_regression_guesser
from restaurant_guest_forecasting.api.models.train_models import train_random_regression_guesser

app = FastAPI()

# Train & save models 
train_random_regression_guesser()

@app.get("/")
def read_root():
    return {"Hello": "World;)"}


@app.post("/predict_guests/random")
async def predict_guests(features: ModelInput):
    model = load_random_regression_guesser()
    
    input_df = features.to_df()

    try:
        prediction = model.predict(input_df)
        return {"predicted_guests": prediction}
    except Exception as e:
        return {"error": str(e)}

