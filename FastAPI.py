import io
import torch

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from project_name.models.cnn import CNNBackbone
from project_name.models.Preprocessing_class import Preprocessing
from numpy import array


app = FastAPI(title="Depth Prediction API",
              description="Uploads an image and "
                          "returns a predicted depth map.")

# Setup
MODEL_PATH = "cnn_best.pth"
model = CNNBackbone(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
preprocessor = Preprocessing(tile_size=(256, 256))


def process_image(file_bytes: bytes,
                  model: torch.nn.Module,
                  preprocessor: Preprocessing) -> io.BytesIO:
    """Function to process image

    Args:
        file_bytes (bytes): file in byte representation
        model (torch.nn.Module): neural network model
        preprocessor (Preprocessing): preprocessing class

    Returns:
        bytes: depth image in byte format
    """
    img_array = preprocessor.load_image(io.BytesIO(file_bytes))
    tiles = preprocessor.tile_with_padding(img_array)
    depth_tiles = []

    for tile in tiles:
        input_tensor = preprocessor.to_tensor(tile).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        depth_tiles.append(output.squeeze().cpu().numpy())

    depth_map = preprocessor.reconstruct_depth(array(depth_tiles))
    depth_rgb = preprocessor.depth_to_rgb(depth_map, invert=True)

    result_image = Image.fromarray(depth_rgb)
    byte_io = io.BytesIO()
    result_image.save(byte_io, format="PNG")
    byte_io.seek(0)
    return byte_io


@app.post("/predict_depth/", summary="Predict depth from image")
async def predict_depth(file: UploadFile = File(...)) -> StreamingResponse:
    """Function to generate depth from image given

    Args:
        file (UploadFile, optional): image file that is uploaded.
        Defaults to File(...).

    Raises:
        HTTPException: image format is not supported
        HTTPException: when there is an error processing the image

    Returns:
        StreamingResponse: response with processed image
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image format")
    try:
        contents = await file.read()
        image_bytes = process_image(contents, model, preprocessor)
        return StreamingResponse(image_bytes, media_type="image/png")
    except Exception:
        raise HTTPException(status_code=500, detail="Error processing image.")


@app.get("/", summary="Health check")
def read_root() -> dict:
    """Health check function

    Returns:
        dict: dict of the status
    """
    return {"status": "healthy"}
