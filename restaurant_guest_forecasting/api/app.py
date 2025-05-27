from fastapi import FastAPI
from input import ModelInput


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
    # prediction = 

