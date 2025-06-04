import os
from fastapi import FastAPI, HTTPException

from sklearn.linear_model import LinearRegression

from restaurant_guest_forecasting.api.input import ModelInput
from restaurant_guest_forecasting.models.utils.load_models import \
    load_model, load_mlp
from restaurant_guest_forecasting.models.mlp.mlp import MultiTaskMLP
from restaurant_guest_forecasting.models.utils.train_base_model import \
    train_and_save_model
from restaurant_guest_forecasting.models.utils.evaluate_models import \
    test_mse, test_mlp_mse
from restaurant_guest_forecasting.models.random_guesser.random_regression_guesser\
      import RandomRegressionGuesser

NEURONS = [37] + [1024]*6 + [512, 256, 128]

def _predict_guests(model: LinearRegression | RandomRegressionGuesser, 
                   input: ModelInput):
    if input.is_valid():
        input_df = input.to_df()
        return model.predict(input_df) 
    raise RuntimeError(input.invalid_reason)


app = FastAPI()

# Train & save models 
train_and_save_model(RandomRegressionGuesser(),
                      "random_regression_guesser.pkl")

train_and_save_model(LinearRegression(),
                      "linear_regression.pkl")





@app.get("/")
async def read_root():
    try:
        with open(os.path.join(os.path.dirname(__file__), "welcome.txt"), "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        return {"Welcome message": lines}
    except Exception:
        return {"Welcome message": "Welcome to the Restaurant Guest \
                Forecasting API!"}
    


@app.get("/predict_guests/random/eval")
async def random_guest_eval():
    model = load_model("random_regression_guesser.pkl")
    return {"random_guesser_test_mse": f"{test_mse(model):.2f}"}


@app.get("/predict_guests/model/eval")
async def linear_regression_guest_eval():
    model = load_model("linear_regression.pkl")
    return {"model_test_mse":  f"{test_mse(model):.2f}"}


@app.get("/predict_guests/compare")
async def linear_regression_guest_eval_compare():
    random_guesser = load_model("random_regression_guesser.pkl")
    model          = load_model("linear_regression.pkl")
    mlp            = load_mlp(NEURONS)
    return {"random_guess_test_mse": f"{test_mse(random_guesser):.2f}",
            "model_test_mse": f"{test_mse(model):.2f}",
            "mlp_test_mse": f"{test_mlp_mse(mlp):.2f}"}


@app.post("/predict_guests/random")
async def predict_guests_rand(input: ModelInput):
    model = load_model("random_regression_guesser.pkl")
    try:
        prediction = _predict_guests(model, input)
        return {"predicted_guests": f"{prediction[0]:.2f}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_guests/model")
async def predict_guests_model(input: ModelInput):
    model = load_model("linear_regression.pkl")
    try:
        prediction = _predict_guests(model, input)
        return {"predicted_guests": f"{prediction[0]:.2f}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
