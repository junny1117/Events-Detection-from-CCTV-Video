import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import time

sys.path.append(os.path.join(os.getcwd(), 'yolov5'))
sys.path.append(os.path.join(os.getcwd(), 'yolov5', 'utils'))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox

class ObjectDetector:
    def __init__(self, weights_path, roi_intrusion, roi_no_parking, intrusion_time_threshold, loitering_time_threshold=10, parking_time_threshold=10):
        device = select_device('0' if torch.cuda.is_available() else 'cpu')

        if isinstance(weights_path, (Path, str)):
            weights_path = str(weights_path)
        
        self.model = DetectMultiBackend(weights_path, device=device)
        self.model.warmup(imgsz=(1, 3, 640, 640))

        self.roi_intrusion = roi_intrusion
        self.roi_no_parking = roi_no_parking
        self.intrusion_time_threshold = intrusion_time_threshold
        self.loitering_time_threshold = loitering_time_threshold
        self.parking_time_threshold = parking_time_threshold  # 불법 주차 감지 시간 설정
        self.tracked_objects = defaultdict(lambda: {'start_time': 0, 'last_seen': 0})

    def detect_and_draw(self, frame):
        img = letterbox(frame, new_shape=(640, 640))[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.model.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)
        pred = non_max_suppression(pred)

        detected_events = []
        current_time = time.time()

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    class_name = self.model.names[int(cls)]
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2

                    # 배회 감지 로직
                    if class_name == 'person':
                        object_id = f"{int(center_x)}-{int(center_y)}"
                        if object_id in self.tracked_objects:
                            time_in_roi = current_time - self.tracked_objects[object_id]['start_time']
                            if time_in_roi > self.loitering_time_threshold:
                                detected_events.append({'type': '배회', '신뢰도': conf.item()})
                                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 3)
                                cv2.putText(frame, 'Loitering', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        else:
                            self.tracked_objects[object_id] = {'start_time': current_time}

                    # 침입 감지 로직
                    if class_name == 'person' and \
                        self.roi_intrusion[0] < center_x < self.roi_intrusion[0] + self.roi_intrusion[2] and \
                        self.roi_intrusion[1] < center_y < self.roi_intrusion[1] + self.roi_intrusion[3]:
                        detected_events.append({'type': '침입', '신뢰도': conf.item()})
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2])), (int(xyxy[3])), (0, 0, 255), 3)
                        cv2.putText(frame, 'Intrusion', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    # 불법 주차 감지 로직
                    elif class_name == 'car' and \
                        self.roi_no_parking[0] < center_x < self.roi_no_parking[0] + self.roi_no_parking[2] and \
                        self.roi_no_parking[1] < center_y < self.roi_no_parking[1] + self.roi_no_parking[3]:
                        car_id = f"{int(center_x)}-{int(center_y)}"
                        if self.tracked_objects[car_id]['start_time'] == 0:
                            self.tracked_objects[car_id]['start_time'] = current_time
                        time_in_no_parking = current_time - self.tracked_objects[car_id]['start_time']
                        if time_in_no_parking > self.parking_time_threshold:
                            detected_events.append({'type': '불법주차', '신뢰도': conf.item()})
                            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 3)
                            cv2.putText(frame, 'Illegal Parking', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        self.tracked_objects[car_id]['last_seen'] = current_time
                    else:
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2])), (int(xyxy[3])), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name} {conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 오래된 트래킹 객체 제거 (불법 주차, 배회 감지 로직)
        self.tracked_objects = {k: v for k, v in self.tracked_objects.items() if current_time - v['last_seen'] <= max(self.loitering_time_threshold, self.parking_time_threshold)}

        return frame, detected_events
