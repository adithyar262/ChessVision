import torch
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load a chessboard image
img = cv2.imread('/mnt/c/Users/tel23/OneDrive/Documents/Codes/ChessVision/LiveChess2FEN_IMPL/images/test1.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run inference
results = model(img_rgb)

# Visualize results
results.print()
results.show()

# Extract detections
detections = results.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max, confidence, class]