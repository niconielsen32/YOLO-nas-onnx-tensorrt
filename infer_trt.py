import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from numpy import random



from exec_backends.trt_loader import TrtModelNMS
# from models.models import Darknet


def letterbox(img, new_shape=(640,640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

class YOLOR(object):
    def __init__(self, 
            model_weights = 'C:/Users/USER/yolonas/yolo_nas_l_new.trt',
            max_size = 640, 
            names = 'C:/Users/USER/yolonas/Yolo-TensorRT/YOLO-NAS/coco.names'):
        # self.names = [f"tattoo{i}" for i in range(80)]
        self.names = load_classes(names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.imgsz = (max_size, max_size)
        # Load model
        self.model = TrtModelNMS(model_weights, max_size)
        print(self.model)


    def detect(self, bgr_img):
        # input, (x_ratio, y_ratio) =  preprocess(bgr_img, (416, 416))
        # print(input.shape)   
        # Prediction
        ## Padded resize
        h, w, _ = bgr_img.shape
        # bgr_img, _, _ = letterbox(bgr_img)
        frame_resized = cv2.resize(bgr_img, (640, 640))
        image_bchw = np.transpose(np.expand_dims(frame_resized, 0), (0, 3, 1, 2))
        # print(inp.shape)  
        # print(x_ratio, y_ratio)
        ## Inference

        print(image_bchw[0].transpose(0, 1, 2).shape)
        cv2.imwrite("test.jpg", image_bchw[0].transpose(1, 2, 0))
       
        results = self.model.run(image_bchw)
      
        [flat_predictions] = results


        for (sample_index, x1, y1, x2, y2, class_score, class_index) in flat_predictions[flat_predictions[:, 0] == 0]:
            class_index = int(class_index)
            if class_score < 0.4:
                continue
            # Scale the bounding box coordinates back to the original image size
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw class index and confidence
            label = f"{self.names[class_index]}: {class_score:.2f}"
            cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite('result.jpg', frame_resized)
            return frame_resized

if __name__ == '__main__':
    model = YOLOR(model_weights="C:/Users/USER/yolonas/yolo_trt.engine")
    img = cv2.imread('C:/Users/USER/yolonas/test.jpg')
    model.detect(img)

