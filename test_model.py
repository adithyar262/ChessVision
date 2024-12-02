import os
import glob
import torch
import cv2
import numpy as np
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

def detect_chess_pieces(weights_path, img_size, conf_thres, iou_thres, device, source_folder):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights_path, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    img_size = check_img_size(img_size, s=stride)

    # Process images
    for img_path in glob.glob(os.path.join(source_folder, '*')):
        # Read and preprocess image
        img0 = cv2.imread(img_path)
        img = letterbox(img0, img_size, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]

        # Inference
        pred = model(img, augment=False, visualize=False)

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)

        # Process detections
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

                # Draw bounding boxes and labels
                annotator = Annotator(img0, line_width=3, example=str(names))
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

        # Display result
        cv2.imshow(f'Detections in {os.path.basename(img_path)}', img0)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    weights_path = 'runs/train/exp2/weights/best.pt'  # Path to your trained weights
    img_size = 416
    conf_thres = 0.25
    iou_thres = 0.45
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # cuda device, i.e. 0 or 0,1,2,3 or cpu
    source_folder = 'test_images'  # Folder containing your test images

    detect_chess_pieces(weights_path, img_size, conf_thres, iou_thres, device, source_folder)
