from fastapi import FastAPI, HTTPException

from restaurant_guest_forecasting.api.input import ModelInput
from restaurant_guest_forecasting.api.models.load_models import load_random_regression_guesser
from restaurant_guest_forecasting.api.models.train_models import train_random_regression_guesser

app = FastAPI()

# Train & save models 
train_random_regression_guesser()

@app.get("/")
def read_root():
    return {"Hello": "World;)"}

@app.get("/predict_guests/random/eval")
def random_guest_eval():
    model = load_random_regression_guesser()
    return {"random_guesser_train_mse": model.train_mse}


@app.post("/predict_guests/random")
async def predict_guests(input: ModelInput):
    model = load_random_regression_guesser()
    
    if input.is_valid():
        input_df = input.to_df()
        try:
            prediction = model.predict(input_df)
            return {"predicted_guests": prediction}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=400, detail=str(input.invalid_reason))

