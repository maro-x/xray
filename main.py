from fastapi import FastAPI
import base64
import io
from infer import infere
from pydantic import BaseModel
from PIL import Image

app = FastAPI()

class RequestData(BaseModel):
    img_data: str

# -----------------------------
# Route for ResNet18
# -----------------------------
@app.post("/predict/resnet18")
async def predict_res18(req: RequestData):
    img_data = io.BytesIO(base64.b64decode(req.img_data))
    img = Image.open(img_data)
    pred = infere(img, "resnet18")
    return {"model": "resnet18", "pred": pred}

# -----------------------------
# Route for ResNet34
# -----------------------------
@app.post("/predict/resnet34")
async def predict_res34(req: RequestData):
    img_data = io.BytesIO(base64.b64decode(req.img_data))
    img = Image.open(img_data)
    pred = infere(img, "resnet34")
    return {"model": "resnet34", "pred": pred}

# -----------------------------
# Route for MobileNetV2
# -----------------------------
@app.post("/predict/mobilenet")
async def predict_mobilenet(req: RequestData):
    img_data = io.BytesIO(base64.b64decode(req.img_data))
    img = Image.open(img_data)
    pred = infere(img, "mobilenet")
    return {"model": "mobilenet", "pred": pred}
