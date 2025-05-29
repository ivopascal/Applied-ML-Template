import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from sudoku_digitalisation.features.sudoku_preprocessing import SudokuPreprocessor
from sudoku_digitalisation.models.CNN import CNN

app = FastAPI()

# Constants
MODEL_PATH = "sudoku_cnn.keras"
OUTPUT_SIZE = 252


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


@app.post("/predict/")
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