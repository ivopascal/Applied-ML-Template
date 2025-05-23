from fastapi import FastAPI
from input import ModelInput

# from restaurant_guest_forecasting.models.base.linear_regression import load_linear_regression

app = FastAPI()
# model = load_linear_regression()


@app.get("/")
def read_root():
    return {"Hello": "World"}


# async def predict_guests(features: ModelInput):
#     prediction = 