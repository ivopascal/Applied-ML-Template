import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from sudoku_digitalisation.features.sudoku_preprocessing import SudokuPreprocessor
from sudoku_digitalisation.models.CNN import CNN
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title = "Sudoku digitizer",
    summary = "an API endpoint to take in a sudoku and give the corresponding 9x9 representing the sudoku",
    description = """
# An API endpoint to access a CNN
# Model usage
The CNN was trained on a combination of digits in different fonts and some handwritten.
It takes these images by cropping sudokus that were fed to it.
The API takes in an already cropped image of a sudoku and returns a 9x9 matrix representing the numbers inside the sudoku, 0 means an empty square.

## Limitations
The model cannot predict the small numbers that are used in sudokus, 
however these should not impact the prediction.""",
version = "alpha"
)

# Constants
MODEL_PATH = "sudoku_cnn"
OUTPUT_SIZE = 252

class SudokuPredictions(BaseModel):
    predictions: List[List[int]]


def load_model(model_path=MODEL_PATH, output_size=OUTPUT_SIZE):
    print("Loading pre-trained model...")
    sudoku_height = output_size // 9
    cnn = CNN(input_shape=(sudoku_height, sudoku_height, 1), num_classes=10)
    cnn.load(model_path)
    return cnn


def predict_sudoku(cnn, image: Image.Image, output_size=OUTPUT_SIZE):
    print("Preprocessing and predicting...")
    preprocessor = SudokuPreprocessor(clip_limit=3, output_size=output_size)
    _, digit_dataset = preprocessor.sudoku_preprocessing(image)

    predictions = cnn.predict(digit_dataset)
    sudoku_labels = []
    dimension = int(np.sqrt(len(digit_dataset)))

    for i in range(dimension):
        row = []
        for j in range(dimension):
            label = int(np.argmax(predictions[j + (dimension * i)]))
            row.append(label)
        sudoku_labels.append(row)

    return sudoku_labels


# Load the model once when FastAPI starts
cnn_model = load_model()


@app.post("/predict/", description = "Sudoku digitizer endpoint. Upload picture of already cropped sudoku."
                                    "Picture has to be .png, .jpg or .jpeg."
                                    "Returns 9x9 matrix representing the uploaded sudoku with 0 being an empty space.",
                        response_model = SudokuPredictions,
                        response_description = "digitised version of uploaded sudoku, in the form of a 9x9 matrix.")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only image files (.png, .jpg, .jpeg) are accepted")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("L")  # convert to grayscale

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

    try:
        result = predict_sudoku(cnn_model, image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return JSONResponse(content={"sudoku_grid": result})

# Run with: uvicorn fastAPI:app --reload