import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("models/best.pt")

# # Train the model on your dataset
# model.train(
#     data="/mnt/c/Users/tel23/OneDrive/Documents/Codes/ChessVision/"
#          "LiveChess2FEN_IMPL/data/dataset/ChessPiecesDetection/data.yaml",
#     epochs=100,            # Number of epochs
#     imgsz=640,             # Image size for training
#     device="0",            # Index of GPU to use, "cpu" for CPU
# )
#
# # Evaluate model performance on the validation set
# metrics = model.val()

# Perform object detection on an image
results = model("../images/test1.jpg")
results[0].show()

# # Export the model to ONNX format
# path = model.export(format="onnx")