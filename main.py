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

model = YOLO("yolov26m.pt")

