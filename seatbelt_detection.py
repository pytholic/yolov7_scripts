import argparse
import time
import os
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from undistortion import *

# Belt array for batch predictions
belt_pred_left = []
belt_pred_right = []
        
def print_text(img, text: str, org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0,255,0), thickness=2):
    cv2.putText(img, text, org=org, fontFace=fontFace, fontScale=fontScale, color=color, thickness=thickness)

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

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

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
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
        print(len(pred))
        t2 = time_synchronized()
        
        end=time.time()
        fps = 1 / (end - start)
        
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            #im0 = restore_distorted_frame(im0)
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                #############################################################################
                
                # Arrays to hold belt prediction for each frame
                pred_left = []
                pred_right = [] 
                
                #print(det[:, -1])
                #print(det[:, :4])
                #print(im0.shape)
                
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
                    
                    if len(belts) == 0:
                        pred_left.append('Not detected')
                        pred_right.append('Not detected')
                    elif x1 > img_center:
                        pred_right.append('Detected')
                    elif x1 < img_center:
                        pred_left.append('Detected')

                # print(pred_left)
                # print(pred_right)

                # Append results in batch arrays
                if len(pred_left) > 0:
                    belt_pred_left.append('Detected')
                else:
                    belt_pred_left.append('Not detected')
                    
                if len(pred_right) > 0:
                    belt_pred_right.append('Detected')
                else:
                    belt_pred_right.append('Not detected')
   
                # Pop first element when batch is greater than 200
                if len(belt_pred_left) > 200:
                    belt_pred_left.pop(0)
                if len(belt_pred_right) > 200:
                    belt_pred_right.pop(0)
                
                # Threshold logic left
                cnt_left_on = belt_pred_left.count("Detected")
                cnt_left_off = belt_pred_left.count("Not detected")
                
                thres_left = cnt_left_on / (cnt_left_on + cnt_left_off)
                
                if thres_left > 0.5:
                    print_text(im0, "Left belt is on", org=(100,200))
                else:
                    print_text(im0, "Left belt is off", org=(100,200))

                # Threshold logic right
                cnt_right_on = belt_pred_right.count("Detected")
                cnt_right_off = belt_pred_right.count("Not detected")
                
                print(f"Count left: ON: {cnt_left_on}, OFF: {cnt_left_off}")
                print(f"Count Right: ON: {cnt_right_on}, OFF: {cnt_right_off}")
                
                thres_right = cnt_right_on / (cnt_right_on + cnt_right_off)
                
                if thres_right > 0.5:
                    print_text(im0, "Right belt is on", org=(1200,200))
                    #print("Belt is on.")
                else:
                    print_text(im0, "Right belt is off", org=(1200,200))
                
                #############################################################################3

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                
            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
            # Stream results
            if view_img:
                #result = np.concatenate((im0, img), axis=1)
                #cv2.namedWindow(str(p), cv2.WINDOW_NORMAL)
                #im0 = cv2.resize(im0, (img.shape[3], img.shape[2]))
                #print(im0.shape)
                #cv2.putText(im0, f"FPS: {fps:.2f}", org=(50,50), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,255), thickness=2)
                print_text(im0, f"FPS: {fps:.2f}", org=(50,50), fontScale=1.5, color=(0,0,255), thickness=2)
                cv2.imshow(str(p), im0)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                
                #cv2.imshow(str(p), im0)
                #cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w, h = im0.shape[1], im0.shape[0]
                            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[0], im0.shape[1]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
