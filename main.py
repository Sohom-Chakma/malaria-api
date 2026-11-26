import logging
from io import BytesIO
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO
from inference import (
model, predict, MODEL_ID, IMAGE_SIZE,
CONFIDENCE, MAX_DETECTIONS
)

app = FastAPI(title="YOLOv26m Malaria Detection API", version="1.0.0")
log = logging.getLogger("uvicorn.error")
FORMAT = {"JPEG", "JPG", "PNG","TIFF"}

@app.get("/models")
def models():
    return {"models": [{
        "model_id": MODEL_ID,
        "imgsz": IMAGE_SIZE,
        "confidence_threshold": CONFIDENCE,
        "max_detections": MAX_DETECTIONS,
        "classes": model.names,
        "device": "cpu",
    }]}

def decode(file):
    raw = file.read()
    with Image.open(BytesIO(raw)) as image:
        if image.format not in FORMAT:
            raise HTTPException(status_code=400, detail=f"Unsupported image format. Use JPEG, JPG, PNG, or TIFF: {image.format}")

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
def predict_images(files: list[UploadFile] = File(...)):

    results = []
    
    for index, file in enumerate(files, start = 1):
        item = {"index": index, "filename": file.filename}
        try:
            image = decode(file)
        except (ValueError, OSError, SyntaxError):
            results.append(error_item(item, "invalid_image", "Unsupported or invalidimage format."))
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


                