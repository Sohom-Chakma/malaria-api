import logging
from io import BytesIO
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from typing import Annotated
from pydantic import WithJsonSchema
from inference import (
model, predict, MODEL_ID, IMAGE_SIZE,
CONFIDENCE_THRESHOLD, MAX_DETECTIONS
)

app = FastAPI(title="YOLOv26m Malaria Detection API", version="1.0.0")
log = logging.getLogger("uvicorn.error")
FORMAT = {"JPEG", "PNG", "TIFF"}

BinaryUploadFile = Annotated[
    UploadFile,
    WithJsonSchema({"type": "string", "format": "binary"})
]

@app.get("/models")
def models():
    return {"models": [{
        "model_id": MODEL_ID,
        "imgsz": IMAGE_SIZE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "max_detections": MAX_DETECTIONS,
        "classes": model.names,
        "device": "cpu",
    }]}

def decode(file):
    raw = file.file.read()

    with Image.open(BytesIO(raw)) as image:
        if image.format not in FORMAT:
            raise ValueError(
                f"Unsupported image format: {image.format}"
            )

        return image.convert("RGB")

def error_item(item, code, message):
    return { 
        **item, 
            "status": "error",
            "error": {
                "code": code, 
                "message": message
                }
            }

@app.post("/predict")
def predict_images(files: list[BinaryUploadFile] = File(...)):

    results = []
    
    for index, file in enumerate(files, start = 1):
        item = {"index": index, "filename": file.filename}
        try:
            image = decode(file)
        except (ValueError, OSError, SyntaxError, Image.DecompressionBombError):
            results.append(error_item(item, "invalid_image", "Unsupported or invalid image format."))
            continue

        try:
            with image:
                output = predict(image)
                results.append({**item, "status": "ok", "output": output})
                
        except Exception:
            log.exception("Inference failed for item %d", index)
            results.append(error_item(item, "inference_failed", "Inference failed for this image."
            ))

    succeeded = sum(1 for result in results if result["status"] == "ok")
    return {
        "model_id": MODEL_ID,
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "results": results
    }


                