from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/prediction")
async def create_prediction(file: UploadFile = File(...)):
    print("Received file:", file.filename)
    # send the file to the model and get the prediction
    prediction = "some prediction"
    return {"filename": file.filename, "prediction": prediction}
