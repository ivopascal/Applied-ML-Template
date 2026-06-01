from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import models, transforms
import io

CLASSES = (
    "Hernia",
    "Pneumonia",
    "Fibrosis",
    "Effusion",
    "Edema",
    "Emphysema",
    "Mass",
    "Nodule",
    "Atelectasis",
    "Cardiomegaly",
    "Infiltration",
    "Pleural_Thickening",
    "Consolidation",
    "Pneumothorax",
    "No Finding",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = r"C:\AML\AML\chest_xray\interfaces\backend\api\densenet_pretrained_epoch10.pth"


def create_model():
    model = models.densenet161(weights=None)

    # Replace classifier to match your number of classes
    model.classifier = torch.nn.Linear(
        model.classifier.in_features,
        len(CLASSES)
    )

    # Replace first convolution layer for grayscale images
    old_first_layer = model.features[0]

    new_first_layer = torch.nn.Conv2d(
        in_channels=1,
        out_channels=old_first_layer.out_channels,
        kernel_size=old_first_layer.kernel_size,
        stride=old_first_layer.stride,
        padding=old_first_layer.padding,
        bias=old_first_layer.bias is not None,
    )

    model.features[0] = new_first_layer

    return model

model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

@app.post("/prediction")
async def create_prediction(file: UploadFile = File(...)):
    print("Received file:", file.filename)

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.sigmoid(outputs)[0]

    all_predictions = []

    for class_name, probability in zip(CLASSES, probabilities):
        all_predictions.append({
            "label": class_name,
            "confidence": float(probability.item())
        })

    predictions = sorted(
        all_predictions,
        key=lambda item: item["confidence"],
        reverse=True
    )

    return {
        "filename": file.filename,
        "predictions": predictions,
    }