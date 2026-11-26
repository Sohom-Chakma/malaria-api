from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import uuid
from io import BytesIO

app = FastAPI(title="YOLOv26m Malaria Detection API", version="1.0.0")

model = YOLO("yolov26m.pt")

