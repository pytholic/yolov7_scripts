import argparse
import time
import os, glob
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from undistortion import *


def print_text(img, text: str, org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0,255,0), thickness=2):
    cv2.putText(img, text, org=org, fontFace=fontFace, fontScale=fontScale, color=color, thickness=thickness)


def detect(frame, save_img=False):
    weights, imgsz = opt.weights, opt.img_size

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Load image and preprocess
    im0s = frame
    img = letterbox(im0s)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    idx = names.index('handbag')
    names[idx] = 'seatbelt'
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    start = time.time() 
        
    img = torch.from_numpy(img).to(device)
    #print(img.shape)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

        #print(img.shape)

    # Warmup
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment=opt.augment)[0]  
         
    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]
    t2 = time_synchronized()
    
    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t3 = time_synchronized()

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0 = '', im0s

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            ###SEATBELT LOGIC START###
            
            # Arrays to hold belt prediction for each frame
            pred_left = []
            pred_right = [] 
            
            # Finding index for belt detection in output array
            idx = np.where(det[:, -1].cpu() == 26)
            idx = np.asarray(idx).squeeze(axis=0)
            
            belts = {}
            for i in idx:
                #belt_label = det[:, -1][i]
                belt_bbox = det[:, :4][i]
                belts[f'{i}'] = belt_bbox
       
            # Checking left and right belt for current frame
            for belt in belts.values():
                x1 = belt[0]
                img_center = int(im0.shape[1] // 2)
                if x1 > img_center:
                    pred_right.append('Detected')
                elif x1 < img_center:
                    pred_left.append('Detected')
                
            ###SEATBELT LOGIC END###

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # Add bbox to image
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                
        # Print time (inference + NMS)
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')        

        return im0, pred_left, pred_right



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-w6.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')

    opt = parser.parse_args()
    print(opt)

    VIDEO = './test_videos/3.mp4'
    cap = cv2.VideoCapture(VIDEO)
    while True:
        with torch.no_grad():
            _, image = cap.read()

            img, pred_left, pred_right = detect(frame=image)

            print(f"Prediction left : {pred_left}")
            print(f"Prediction right : {pred_right}")

            # Show results
            cv2.imshow('Results', img)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows() 


    #image = cv2.imread('/home/pytholic/Desktop/Projects/icms_data/data/images/4/0400002.jpg')
    #test_dir = '/home/pytholic/Desktop/Projects/icms_data/data/images/4/'
    # with torch.no_grad():
    #     for image_path in glob.glob(test_dir + '*.jpg'):
    #         image = cv2.imread(image_path)

    #         img, pred_left, pred_right = detect(frame=image)

    #         print(f"Prediction left : {pred_left}")
    #         print(f"Prediction right : {pred_right}")

    #         # Show results
    #         cv2.imshow('Results', img)
    #         key = cv2.waitKey(0)
    #         if key == 27:
    #             cv2.destroyAllWindows() 
