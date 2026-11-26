from pathlib import Path
from threading import Lock
from time import perf_counter

from PIL import Image
from ultralytics import YOLO

MODEL_ID = "malaria-yolov26m"
WEIGHTS = Path(__file__).with_name("yolo_model.pt")
IMAGE_SIZE = 1280
CONFIDENCE_THRESHOLD = 0.25
MAX_DETECTIONS = 50

model = YOLO(WEIGHTS), task = "detect"