import sys
import os
import cv2
import torch
import datetime
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, session
from flask_socketio import SocketIO, emit
from pathlib import Path
from python.video_stream import VideoStream
from python.detection import ObjectDetector
from python.database import init_db, db_session, Event
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app)

# Dummy credentials for login
USERNAME = 'admin'
PASSWORD = '1234'

# rtsp_url = "rtsp://username:password@ip_address:port/path"
# video_stream = VideoStream(rtsp_url)
video_stream = VideoStream("test.mp4")
#video_capture = cv2.VideoCapture(0)

weights_path = 'best.pt'
roi_intrusion = (100, 200, 300, 400)
roi_no_parking = (400, 500, 300, 400)
intrusion_time_threshold = 5
detector = ObjectDetector(weights_path, roi_intrusion, roi_no_parking, intrusion_time_threshold)

tracked_objects = {}

@app.before_request
def setup():
    init_db()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid Credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    global roi_intrusion, roi_no_parking
    if request.method == 'POST':
        roi_intrusion = (
            int(request.form['intrusion_x']),
            int(request.form['intrusion_y']),
            int(request.form['intrusion_width']),
            int(request.form['intrusion_height'])
        )
        roi_no_parking = (
            int(request.form['no_parking_x']),
            int(request.form['no_parking_y']),
            int(request.form['no_parking_width']),
            int(request.form['no_parking_height'])
        )
        detector.roi_intrusion = roi_intrusion  # ObjectDetector 인스턴스의 ROI 업데이트
        detector.roi_no_parking = roi_no_parking  # ObjectDetector 인스턴스의 ROI 업데이트
        return redirect(url_for('index'))
    return render_template('settings.html')

@app.route('/select_roi')
@login_required
def select_roi():
    global roi_intrusion, roi_no_parking
    frame = video_stream.get_frame()

    # 프레임 크기 조정 (예: 25%로 축소)
    scale_percent = 50 # 원하는 크기 비율로 변경 (예: 25%)
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # 침입 감지 영역 선택
    roi_intrusion_resized = cv2.selectROI("Select Intrusion ROI", resized_frame, fromCenter=False, showCrosshair=True)
    roi_intrusion = (
        int(roi_intrusion_resized[0] / scale_percent * 100),
        int(roi_intrusion_resized[1] / scale_percent * 100),
        int(roi_intrusion_resized[2] / scale_percent * 100),
        int(roi_intrusion_resized[3] / scale_percent * 100)
    )
    cv2.destroyWindow("Select Intrusion ROI")

    # 불법 주차 영역 선택
    roi_no_parking_resized = cv2.selectROI("Select No Parking ROI", resized_frame, fromCenter=False, showCrosshair=True)
    roi_no_parking = (
        int(roi_no_parking_resized[0] / scale_percent * 100),
        int(roi_no_parking_resized[1] / scale_percent * 100),
        int(roi_no_parking_resized[2] / scale_percent * 100),
        int(roi_no_parking_resized[3] / scale_percent * 100)
    )
    cv2.destroyWindow("Select No Parking ROI")

    detector.roi_intrusion = roi_intrusion  # ObjectDetector 인스턴스의 ROI 업데이트
    detector.roi_no_parking = roi_no_parking  # ObjectDetector 인스턴스의 ROI 업데이트

    return redirect(url_for('index'))

@app.route('/events')
@login_required
def events():
    events = Event.query.order_by(Event.timestamp.desc()).all()  # 최신순 정렬
    return render_template('events.html', events=events)


def gen_frames():
    global tracked_objects
    while True:
        frame = video_stream.get_frame()
        if frame is None:
            break
        #ret, frame = video_capture.read()  # USB 웹캠에서 프레임 읽기
        #if not ret:
            #break

        frame, detected_events = detector.detect_and_draw(frame)

        for event in detected_events:
            new_event = Event(
                label=event['type'],
                confidence=event['신뢰도'],
                timestamp=datetime.datetime.now()
            )
            db_session.add(new_event)
            db_session.commit()
            socketio.emit('new_event', {
                'label': new_event.label,
                'confidence': new_event.confidence,
                'timestamp': new_event.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    socketio.run(app, debug=True)

    #video_capture.release()
    #cv2.destroyAllWindows()
