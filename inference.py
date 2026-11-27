from pathlib import Path
from threading import Lock
from time import perf_counter

from PIL import Image
from ultralytics import YOLO

import base64
from io import BytesIO

MODEL_ID = "malaria-yolov26m"
CONFIDENCE_THRESHOLD = 0.25
MAX_DETECTIONS = 50
IMAGE_SIZE = 1280

model = YOLO(Path(__file__).with_name("yolov26m.pt"))
model_lock = Lock()

def predict(image, return_image=False):
    start = perf_counter()
    with model_lock:
        results = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            max_det=MAX_DETECTIONS,
            device="cpu",
        )[0]

    detections = []
    for detection_id, box in enumerate(results.boxes, start=1):
        class_id = int(box.cls.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "detection_id": detection_id,
            "class_id": class_id,
            "class_name": results.names[class_id],
            "confidence": float(box.conf.item()),
            "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
    })

    height, width = results.orig_shape
    return {
        "image_size": {"height": height, "width": width},
        "detection_count": len( detections ),
        "detections": detections,
        "processing_ms": round( ( perf_counter() - start)  * 1000, 2 )
    }

if __name__ == "__main__":
    import json
    import sys

    for path in sys.argv[1:]:
        with Image.open(path) as image:
          output = predict( image.convert( "RGB" ) )
        print( path )
        print( json.dumps( output, indent=2 ) )